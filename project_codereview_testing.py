import os
import ast
import re
import glob
import json
import sys
import importlib
from typing import Dict, List, Set, Any, Tuple
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    print("Could not import pysqlite3")
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import numpy as np
from chromadb.utils import embedding_functions
import datetime

def patch_chromadb_version_check():
    # This is a hack to bypass ChromaDB's version check
    try:
        import sqlite3
        
        # Directly modify the version check in ChromaDB's __init__
        def mock_sqlite_version_check():
            # Always return True to bypass version check
            return True
        
        # Attempt to monkey patch the version check
        chromadb_init = importlib.import_module('chromadb.__init__')
        chromadb_init.check_sqlite_version = mock_sqlite_version_check
        
        print("ChromaDB SQLite version check patched successfully")
    except Exception as e:
        print(f"Error patching ChromaDB version check: {e}")

# Apply the patch
patch_chromadb_version_check()

class CodeBertEmbeddingFunction:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            text = text[:8000]  # Truncate long texts
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(
                outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()
            )
        return embeddings

class CodeEmbeddingExtractor:
    def __init__(self, project_path: str, db_path: str = "./chromadb"):
        self.project_path = os.path.abspath(project_path)
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.embedding_function = CodeBertEmbeddingFunction(
            model=self.model,
            tokenizer=self.tokenizer
        )
        self.collection = self.client.get_or_create_collection(
            name="code_embeddings",
            embedding_function=self.embedding_function
        )
        self.extensions = {
            ".py": self._extract_python_metadata,
            ".js": self._extract_javascript_metadata,
            ".ts": self._extract_typescript_metadata,
            ".java": self._extract_java_metadata,
            ".cpp": self._extract_cpp_metadata,
            ".c": self._extract_c_metadata,
            ".php": self._extract_php_metadata
        }

    def _detect_language(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        language_map = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp", 
            ".c": "c", 
            ".php": "php"
        }
        return language_map.get(ext, "unknown")

    def _extract_python_metadata(self, content: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(content)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            return {
                "classes": classes,
                "functions": functions,
                "complexity": len(list(ast.walk(tree)))
            }
        except Exception as e:
            return {"error": str(e)}

    def _extract_javascript_metadata(self, content: str) -> Dict[str, Any]:
        # Similar logic to python but using JavaScript parsing methods
        classes = re.findall(r'class\s+(\w+)', content)
        functions = re.findall(r'function\s+(\w+)', content)
        return {
            "classes": classes,
            "functions": functions
        }

    # Similar placeholders for other language extraction methods
    def _extract_typescript_metadata(self, content: str) -> Dict[str, Any]:
        return self._extract_javascript_metadata(content)

    def _extract_java_metadata(self, content: str) -> Dict[str, Any]:
        classes = re.findall(r'class\s+(\w+)', content)
        methods = re.findall(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', content)
        return {
            "classes": classes,
            "methods": methods
        }

    def _extract_cpp_metadata(self, content: str) -> Dict[str, Any]:
        classes = re.findall(r'class\s+(\w+)', content)
        functions = re.findall(r'\w+\s+(\w+)\s*\([^)]*\)\s*{', content)
        return {
            "classes": classes,
            "functions": functions
        }

    def _extract_c_metadata(self, content: str) -> Dict[str, Any]:
        functions = re.findall(r'\w+\s+(\w+)\s*\([^)]*\)\s*{', content)
        return {
            "functions": functions
        }

    def _extract_php_metadata(self, content: str) -> Dict[str, Any]:
        classes = re.findall(r'class\s+(\w+)', content)
        functions = re.findall(r'function\s+(\w+)', content)
        return {
            "classes": classes,
            "functions": functions
        }

    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.extensions:
            return self.extensions[ext](content)
        return {}

    def _should_ignore(self, file_path: str) -> bool:
        ignore_patterns = [
            r'\.git/',
            r'\.venv/',
            r'__pycache__/',
            r'node_modules/',
            r'dist/',
            r'build/',
            r'\.idea/',
            r'\.vscode/'
        ]
        for pattern in ignore_patterns:
            if re.search(pattern, file_path):
                return True
        return False

    # Rest of the methods remain the same as in your original code...
    # (extract_and_store, retrieve_related_files methods)
    def extract_and_store(self) -> None:
        file_count = 0
        skip_count = 0
        print(f"Scanning project directory: {self.project_path}")
        for root, _, files in os.walk(self.project_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_path)
                if self._should_ignore(rel_path):
                    skip_count += 1
                    continue
                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext not in self.extensions:
                        skip_count += 1
                        continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content) > 1024 * 1024:
                        print(f"Skipping large file: {rel_path}")
                        skip_count += 1
                        continue
                    existing = self.collection.get(ids=[rel_path])
                    if existing["ids"]:
                        print(f"Skipping existing file: {rel_path}")
                        skip_count += 1
                        continue
                    metadata = self._extract_metadata(file_path, content)
                    metadata["file_name"] = file
                    metadata["path"] = rel_path
                    metadata["language"] = self._detect_language(file_path)
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            metadata[key] = ",".join(value)
                    self.collection.add(
                        documents=[content],
                        metadatas=[metadata],
                        ids=[rel_path]
                    )
                    file_count += 1
                    if file_count % 20 == 0:
                        print(f"Processed {file_count} files...")
                except Exception as e:
                    print(f"Error processing {rel_path}: {str(e)}")
                    skip_count += 1
        print(f"Completed processing {file_count} files. Skipped {skip_count} files.")

    def retrieve_related_files(self, file_path: str, limit: int = 3) -> List[Dict[str, Any]]:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            content = content[:8000] if len(content) > 8000 else content
            results = self.collection.query(
                query_texts=[content],
                n_results=limit + 1
            )
            related_files = []
            rel_path = os.path.relpath(file_path, self.project_path)
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id == rel_path:
                    continue
                related = {
                    "path": doc_id,
                    "language": self._detect_language(doc_id),
                    "similarity": results["distances"][0][i] if "distances" in results else 0.0
                }
                related_files.append(related)
            return related_files[:limit]
        except Exception as e:
            print(f"Error retrieving related files for {file_path}: {str(e)}")
            return []

class CodeReviewSystem:
    def __init__(self, extractor, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        self.extractor = extractor
        print(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    def _extract_signatures(self, file_path: str) -> str:
        """Extract class and function signatures from a file with line numbers."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            language = self.extractor._detect_language(file_path)
            signatures = []

            if language == "python":
                # Match class and function definitions in Python
                class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
                func_pattern = r'def\s+(\w+)\s*\([^)]*\):'

                for match in re.finditer(class_pattern, content):
                    class_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"class {class_name} (line {line_number})")

                for match in re.finditer(func_pattern, content):
                    func_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"def {func_name}() (line {line_number})")

            elif language in ["javascript", "typescript"]:
                # Match class and function definitions in JS/TS
                class_pattern = r'class\s+(\w+)'
                func_patterns = [
                    r'function\s+(\w+)\s*\(',
                    r'const\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)',
                    r'(\w+)\s*:\s*function',
                    r'(\w+)\s*\([^)]*\)\s*{'
                ]

                for match in re.finditer(class_pattern, content):
                    class_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"class {class_name} (line {line_number})")

                for pattern in func_patterns:
                    for match in re.finditer(pattern, content):
                        func_name = match.group(1)
                        line_number = content[:match.start()].count('\n') + 1
                        signatures.append(f"function {func_name}() (line {line_number})")

            elif language == "java":
                # Match class and method definitions in Java
                class_pattern = r'(?:public|private|protected)?\s+class\s+(\w+)'
                method_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\('

                for match in re.finditer(class_pattern, content):
                    class_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"class {class_name} (line {line_number})")

                for match in re.finditer(method_pattern, content):
                    method_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"method {method_name}() (line {line_number})")

            elif language in ["cpp", "c"]:
                # Match function definitions in C/C++
                func_pattern = r'(\w+)\s+(\w+)\s*\([^)]*\)\s*{'
                class_pattern = r'class\s+(\w+)'

                if language == "cpp":
                    for match in re.finditer(class_pattern, content):
                        class_name = match.group(1)
                        line_number = content[:match.start()].count('\n') + 1
                        signatures.append(f"class {class_name} (line {line_number})")

                for match in re.finditer(func_pattern, content):
                    return_type = match.group(1)
                    func_name = match.group(2)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"{return_type} {func_name}() (line {line_number})")

            elif language == "php":
                # Match class and function definitions in PHP
                class_pattern = r'class\s+(\w+)'
                func_pattern = r'function\s+(\w+)'

                for match in re.finditer(class_pattern, content):
                    class_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"class {class_name} (line {line_number})")

                for match in re.finditer(func_pattern, content):
                    func_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    signatures.append(f"function {func_name}() (line {line_number})")

            # Return the signatures or a message if none found
            if signatures:
                return "\n".join(signatures)
            else:
                return "No class or function signatures found"

        except Exception as e:
            return f"Error extracting signatures: {str(e)}"

    def _extract_imports(self, file_path: str) -> List[str]:
        """Extract import statements from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            language = self.extractor._detect_language(file_path)
            imports = []

            if language == "python":
                # Match import statements in Python
                import_patterns = [
                    r'import\s+([\w.]+)',
                    r'from\s+([\w.]+)\s+import'
                ]

                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        imports.append(match.group(1))

            elif language in ["javascript", "typescript"]:
                # Match import statements in JS/TS
                import_patterns = [
                    r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                    r'require\([\'"]([^\'"]+)[\'"]\)'
                ]

                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        imports.append(match.group(1))

            elif language == "java":
                # Match import statements in Java
                import_pattern = r'import\s+([\w.]+);'

                for match in re.finditer(import_pattern, content):
                    imports.append(match.group(1))

            elif language in ["cpp", "c"]:
                # Match include statements in C/C++
                include_pattern = r'#include\s+[<"]([^>"]+)[>"]'

                for match in re.finditer(include_pattern, content):
                    imports.append(match.group(1))

            elif language == "php":
                # Match use/require/include statements in PHP
                import_patterns = [
                    r'use\s+([\w\\]+)',
                    r'require(_once)?\s+[\'"]([^\'"]+)[\'"]',
                    r'include(_once)?\s+[\'"]([^\'"]+)[\'"]'
                ]

                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        if 'use' in pattern:
                            imports.append(match.group(1))
                        else:
                            imports.append(match.group(2) if match.group(2) else match.group(1))

            return imports

        except Exception as e:
            print(f"Error extracting imports: {str(e)}")
            return []

    def retrieve_code_context(self, file_path: str, num_related_files: int = 3) -> Dict[str, Any]:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            target_code = f.read()

        rel_path = os.path.relpath(file_path, self.extractor.project_path)
        metadata = self.extractor._extract_metadata(file_path, target_code)
        language = self.extractor._detect_language(file_path)

        # Extract imports from the target file
        imports = self._extract_imports(file_path)

        # Get related files based on embeddings
        related_files = self.extractor.retrieve_related_files(file_path, limit=num_related_files)

        # Add line numbering information
        lines = target_code.split('\n')
        line_count = len(lines)

        context = {
            "target_file": {
                "path": rel_path,
                "language": language,
                "metadata": metadata,
                "code": target_code,
                "line_count": line_count,
                "imports": imports
            },
            "related_files": []
        }

        for related in related_files:
            related_path = os.path.join(self.extractor.project_path, related["path"])
            try:
                # Extract signatures instead of full code
                signatures = self._extract_signatures(related_path)

                # Check if this file is imported by the target file
                is_imported = False
                related_basename = os.path.basename(related["path"])
                related_module = os.path.splitext(related_basename)[0]

                for imp in imports:
                    if related_module in imp:
                        is_imported = True
                        break

                context["related_files"].append({
                    "path": related["path"],
                    "language": related["language"],
                    "signatures": signatures,
                    "is_imported": is_imported
                })

            except Exception as e:
                print(f"Error reading related file {related['path']}: {str(e)}")

        return context

    def generate_review_prompt(self, context: Dict[str, Any]) -> str:
        target = context["target_file"]

        # Minimal context for imported and related files
        imported_context = ""
        for idx, imported in enumerate(context.get("related_files", []), 1):
            if imported.get("is_imported", False):
                imported_context += f"""
IMPORTED FILE {idx}: {imported["path"]}
LANGUAGE: {imported["language"]}
IMPORTED ELEMENTS:
{imported.get("signatures", "No signatures found")}
"""

        prompt = f"""<\nim_start\n>system\nYou are an expert code reviewer. Given a code file, identify issues and provide feedback in JSON format.

Important instructions:
1. FOCUS EXCLUSIVELY on the target file's code.
2. Do NOT review or analyze the implementation of imported dependencies.
3. If an import seems critical, you may note a general dependency observation.
4. Identify specific line numbers where issues occur in the TARGET FILE ONLY.
5. Structure your response as valid JSON with these exact keys:
   - "target_file_issues": an array of objects, each containing:
     - "description": a string describing the issue in the target file
     - "line_numbers": an array of integers representing affected line numbers
     - "severity": a string with one of these values only: "Low", "Medium", "High", or "Critical"
     - "impact": an array of strings with one or more of: "Code Quality", "Performance", "Security", "Testing", "Maintainability", "Efficiency"
   - "dependency_observations": an array of objects, each containing:
     - "description": a brief, high-level note about imported dependencies
     - "imported_file": the filename of the dependency
     - "severity": a string with one of these values only: "Low", "Medium", "High", or "Critical"
   - "recommendations": an array of strings with textual recommendations (NO CODE SNIPPETS)

Critically important: ONLY analyze the actual code in the TARGET FILE. Dependency context is for REFERENCE ONLY.
<\im_end\n>
<\im_start\n>user\nI need a code review for the following file:
FILE PATH: {target["path"]}
LANGUAGE: {target["language"]}
### CODE TO REVIEW:
```{target["language"]}
{target["code"]}
```

### IMPORTED DEPENDENCIES:
{imported_context}

Please provide a code review in valid JSON format focusing STRICTLY on the target file's code. Do NOT review implementation details of imported files.<\im_end\n><\im_start\n>assistant\n"""
        return prompt

    # Rest of the methods remain the same as in your original code
    # (review_code, save_review_to_file, review_project, etc.)
    def review_code(self, file_path: str, num_related_files: int = 3) -> str:
        try:
            context = self.retrieve_code_context(file_path, num_related_files)
            prompt = self.generate_review_prompt(context)
            print(f"Generating code review for {file_path}...")
            response = self.pipe(prompt, return_full_text=False)[0]["generated_text"]

            if "<\nim_end\n>" in response:
                response = response.split("<\nim_end\n>")[0].strip()

            # Ensure the response is valid JSON
            try:
                json_response = json.loads(response)

                # Transform old format to new format if needed
                if "issue_description" in json_response and "target_file_issues" not in json_response:
                    rel_path = os.path.relpath(file_path, self.extractor.project_path)

                    # Create a new structure with our expected format
                    new_response = {
                        "target_file_issues": [],
                        "dependency_issues": [],
                        "recommendations": json_response.get("recommendations", [])
                    }

                    # Try to categorize old issues into target vs dependency issues
                    for issue in json_response["issue_description"]:
                        if isinstance(issue, dict):
                            description = issue.get("description", "")
                            file_name = issue.get("file_name", rel_path)
                            line_numbers = issue.get("line_numbers", [])
                            severity = json_response.get("severity", "Medium")
                            impact = json_response.get("impact", ["Code Quality"])

                            if file_name == rel_path:
                                new_response["target_file_issues"].append({
                                    "description": description,
                                    "line_numbers": line_numbers,
                                    "severity": severity,
                                    "impact": impact
                                })
                            else:
                                new_response["dependency_issues"].append({
                                    "description": f"This issue in {file_name} impacts the target file because it may affect functionality",
                                    "imported_file": file_name,
                                    "target_file_line_numbers": [],  # Can't determine without more context
                                    "severity": severity
                                })
                        elif isinstance(issue, str):
                            # If it's just a string description
                            new_response["target_file_issues"].append({
                                "description": issue,
                                "line_numbers": [],
                                "severity": json_response.get("severity", "Medium"),
                                "impact": json_response.get("impact", ["Code Quality"])
                            })

                    json_response = new_response

                response = json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                # If the model didn't return valid JSON, try to extract and format it
                import re
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    try:
                        json_response = json.loads(json_match.group(1))
                        response = json.dumps(json_response, indent=2)
                    except:
                        response = f'{{"error": "Could not parse JSON response", "raw_response": {json.dumps(response)}}}'
                else:
                    response = f'{{"error": "Response is not in JSON format", "raw_response": {json.dumps(response)}}}'

            return response
        except Exception as e:
            return f'{{"error": "Error generating code review: {str(e)}"}}'

    def save_review_to_file(self, file_path: str, review: str, output_dir: str = "./reviews") -> str:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        review_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_review.json")

        with open(review_file, 'w', encoding='utf-8') as f:
            f.write(review)

        return review_file

    def review_project(self, output_file="project_review.json") -> None:
        """Review the entire project and compile results in a single JSON file."""
        file_count = 0
        all_reviews = {}

        # Create review directory if it doesn't exist
        reviews_dir = "./reviews"
        os.makedirs(reviews_dir, exist_ok=True)

        # Find all files to review
        files_to_review = []
        for root, _, files in os.walk(self.extractor.project_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.extractor.project_path)

                if self.extractor._should_ignore(rel_path):
                    continue

                ext = os.path.splitext(file_path)[1].lower()
                if ext not in self.extractor.extensions:
                    continue

                files_to_review.append(file_path)

        # Show progress
        total_files = len(files_to_review)
        print(f"Found {total_files} files to review")

        # Now review each file
        for idx, file_path in enumerate(files_to_review, 1):
            try:
                rel_path = os.path.relpath(file_path, self.extractor.project_path)
                print(f"[{idx}/{total_files}] Reviewing {rel_path}...")

                review = self.review_code(file_path)

                # Save individual review file
                individual_review_file = self.save_review_to_file(file_path, review)

                # Add to comprehensive review
                try:
                    review_json = json.loads(review)
                    all_reviews[rel_path] = review_json
                except json.JSONDecodeError:
                    all_reviews[rel_path] = {"error": "Invalid JSON format", "raw_review": review}

                file_count += 1

            except Exception as e:
                print(f"Error reviewing {rel_path}: {str(e)}")
                all_reviews[rel_path] = {"error": str(e)}

        # Add project metadata
        project_review = {
            "project_name": os.path.basename(self.extractor.project_path),
            "review_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_files_reviewed": file_count,
            "file_reviews": all_reviews
        }

        # Save the comprehensive review
        review_path = os.path.join(reviews_dir, output_file)
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(project_review, f, indent=2)

        print(f"Completed reviewing {file_count} files.")
        print(f"Comprehensive review saved to {review_path}")

def run_code_review(project_path, db_path="./chromadb"):
    try:
        if not os.path.exists(db_path):
            print("Creating new code embedding extractor...")
            extractor = CodeEmbeddingExtractor(project_path, db_path)
            extractor.extract_and_store()
        else:
            print("Loading existing code embedding database...")
            extractor = CodeEmbeddingExtractor(project_path, db_path)

        reviewer = CodeReviewSystem(extractor)
        # Use the project review method which creates a comprehensive report
        reviewer.review_project()
        print("Comprehensive code review completed successfully!")
    except Exception as e:
        print(f"Error in code review process: {str(e)}")

if __name__ == "__main__":
    PROJECT_PATH = "/home/azureuser/entire_project_code_review/project/"  # Update this to your project path
    run_code_review(PROJECT_PATH)
