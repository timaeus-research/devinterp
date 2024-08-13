import os
import subprocess


def generate_docs(source_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the autosummary files
    subprocess.run(["sphinx-apidoc", "-o", output_dir, source_dir, "./src/devinterp/mechinterp", "--force"])

    # Modify the generated files to include automodule directives
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".rst"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    content = f.readlines()
                with open(filepath, "w") as f:
                    for line in content:
                        f.write(line)
                        if line.strip() == ".. automodule::":
                            f.write("    :members:\n")
                            f.write("    :undoc-members:\n")
                            f.write("    :private-members:\n")
                            f.write("    :special-members:\n")
                            f.write("    :inherited-members:\n")
                            f.write("    :show-inheritance:\n")

            if ".tl" in file:
                os.remove(filepath)


source_dir = "../src/devinterp"  # Replace with the path to your project
output_dir = "./source"  # Replace with your desired output directory

generate_docs(source_dir, output_dir)
