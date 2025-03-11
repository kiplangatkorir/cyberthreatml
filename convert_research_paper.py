#!/usr/bin/env python3
"""
Convert the CyberThreat-ML Research Paper HTML to a properly formatted LaTeX document
that can be compiled to PDF with readable mathematical equations.
"""

import re
import os
from html.parser import HTMLParser

# HTML Parser to extract content
class ResearchPaperParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_tag = None
        self.sections = []
        self.current_section = {"title": "", "content": []}
        self.in_title = False
        self.in_abstract = False
        self.in_keywords = False
        self.in_references = False
        self.in_code = False
        self.in_equation = False
        self.in_table = False
        self.table_data = []
        self.current_row = []
        self.current_cell = ""
        self.is_header_row = False
        self.buffer = ""
        self.skip_tags = ["style", "script"]
        self.skip_content = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        
        if tag in self.skip_tags:
            self.skip_content = True
            return
            
        # Track sections
        if tag == "h1":
            self.in_title = True
        elif tag in ["h2", "h3", "h4"]:
            # Save previous section if it exists
            if self.current_section["title"]:
                self.sections.append(self.current_section)
            self.current_section = {"title": "", "content": []}
        
        # Track environment
        elif tag == "div" and "class" in attrs_dict:
            if "abstract" in attrs_dict["class"]:
                self.in_abstract = True
            elif "keywords" in attrs_dict["class"]:
                self.in_keywords = True
            elif "references" in attrs_dict["class"]:
                self.in_references = True
        
        # Track code and equations
        elif tag == "pre":
            self.in_code = True
        elif tag == "p" and "class" in attrs_dict and "math" in attrs_dict["class"]:
            self.in_equation = True
            
        # Track tables
        elif tag == "table":
            self.in_table = True
            self.table_data = []
        elif tag == "tr":
            self.current_row = []
            if self.in_table and not self.table_data:  # First row
                self.is_header_row = True
            else:
                self.is_header_row = False
        elif tag == "th" or tag == "td":
            self.current_cell = ""
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_content = False
            return
            
        # End section tracking
        if tag == "h1":
            self.in_title = False
        
        # End environment tracking
        elif tag == "div":
            if self.in_abstract:
                self.in_abstract = False
            elif self.in_keywords:
                self.in_keywords = False
            elif self.in_references:
                self.in_references = False
        
        # End code and equations
        elif tag == "pre":
            self.in_code = False
        elif tag == "p" and self.in_equation:
            self.in_equation = False
            
        # End table tracking
        elif tag == "table":
            self.in_table = False
            self.current_section["content"].append(("table", self.table_data))
        elif tag == "tr":
            if self.current_row:
                self.table_data.append((self.current_row, self.is_header_row))
        elif tag == "th" or tag == "td":
            self.current_row.append(self.current_cell)
    
    def handle_data(self, data):
        if self.skip_content:
            return
            
        data = data.strip()
        if not data:
            return
            
        # Process titles
        if self.in_title:
            self.current_section["title"] = data
        elif self.current_tag in ["h2", "h3", "h4"]:
            self.current_section["title"] = data
        
        # Process content based on environment
        elif self.in_abstract:
            if self.current_tag != "div":  # Skip the "Abstract" title
                self.current_section["content"].append(("abstract", data))
        elif self.in_keywords:
            if "Keywords" in data:
                clean_data = data.replace("Keywords:", "").strip()
                self.current_section["content"].append(("keywords", clean_data))
        elif self.in_references:
            if self.current_tag == "li":
                self.current_section["content"].append(("reference", data))
        
        # Process code and equations
        elif self.in_code:
            self.current_section["content"].append(("code", data))
        elif self.in_equation:
            # Extract the equation from within \[ \] delimiters
            if "\\[" in data and "\\]" in data:
                equation = data.split("\\[")[1].split("\\]")[0].strip()
                self.current_section["content"].append(("equation", equation))
        
        # Process regular text
        elif self.current_tag == "p" and not self.in_table:
            self.current_section["content"].append(("text", data))
        elif self.current_tag in ["li", "ol", "ul"]:
            self.current_section["content"].append(("list", data))
            
        # Process table cells
        elif self.in_table and (self.current_tag == "th" or self.current_tag == "td"):
            self.current_cell = data

    def get_sections(self):
        # Ensure the final section is appended
        if self.current_section["title"]:
            self.sections.append(self.current_section)
        return self.sections

def html_to_latex(html_file, latex_file):
    # Read HTML content
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML
    parser = ResearchPaperParser()
    parser.feed(html_content)
    sections = parser.get_sections()
    
    # Create LaTeX document
    latex_content = []
    latex_content.append(r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{color}
\usepackage{hyperref}
\usepackage{array}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{float}
\usepackage{booktabs}

\geometry{a4paper, margin=1in}

\lstset{
    basicstyle=\small\ttfamily,
    breaklines=true,
    frame=single,
    showstringspaces=false,
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    commentstyle=\color{green!50!black},
    numbers=left,
    numberstyle=\tiny,
    numbersep=5pt
}

\title{CyberThreat-ML: An Explainable Machine Learning Framework for Real-Time Cybersecurity Threat Detection}
\author{Kiplangat Korir}
\date{\vspace{-5ex}}

\begin{document}

\maketitle
{\color{red}\centering\Large WORK IN PROGRESS - INDEPENDENT PROJECT\\}

\begin{center}
Department of Computer Science\\
korirkiplangat22@gmail.com
\end{center}

""")
    
    # Process sections
    for section in sections:
        title = section["title"]
        level = ""
        
        # Determine section level
        if "background" in title.lower() or "introduction" in title.lower() or title.startswith("1."):
            level = r"\section"
        elif "references" in title.lower():
            level = r"\section*"
        elif "." in title and title[0].isdigit() and title[1] == ".":
            level = r"\subsection"
        elif "." in title and title[0].isdigit() and title[2] == ".":
            level = r"\subsubsection"
        else:
            level = r"\subsection"
        
        # Add section title
        if title and "references" not in title.lower():
            latex_content.append(f"{level}{{{title}}}")
        
        # Process content
        for content_type, content in section["content"]:
            if content_type == "abstract":
                latex_content.append(r"\begin{abstract}")
                latex_content.append(content)
                latex_content.append(r"\end{abstract}")
            
            elif content_type == "keywords":
                latex_content.append(r"\noindent\textbf{Keywords:} " + content)
                latex_content.append("")
            
            elif content_type == "text":
                # Clean up content (remove escaped characters)
                cleaned_content = content.replace("\\", "\\\\")
                cleaned_content = cleaned_content.replace("_", "\\_")
                cleaned_content = cleaned_content.replace("%", "\\%")
                cleaned_content = cleaned_content.replace("&", "\\&")
                cleaned_content = cleaned_content.replace("#", "\\#")
                # Add paragraph
                latex_content.append(cleaned_content)
                latex_content.append("")
            
            elif content_type == "equation":
                latex_content.append(r"\begin{equation}")
                latex_content.append(content)
                latex_content.append(r"\end{equation}")
                latex_content.append("")
            
            elif content_type == "code":
                latex_content.append(r"\begin{lstlisting}")
                latex_content.append(content)
                latex_content.append(r"\end{lstlisting}")
                latex_content.append("")
            
            elif content_type == "list":
                latex_content.append(r"\begin{itemize}")
                latex_content.append(r"\item " + content)
                latex_content.append(r"\end{itemize}")
                latex_content.append("")
            
            elif content_type == "reference":
                if "References" in title:
                    latex_content.append(r"\begin{thebibliography}{99}")
                    latex_content.append(r"\bibitem{ref1} " + content)
                    latex_content.append(r"\end{thebibliography}")
            
            elif content_type == "table":
                table_data = content
                if table_data:
                    # Determine number of columns
                    num_cols = len(table_data[0][0]) if table_data else 0
                    
                    # Generate table header
                    latex_content.append(r"\begin{table}[H]")
                    latex_content.append(r"\centering")
                    latex_content.append(r"\begin{tabular}{|" + "c|" * num_cols + "}")
                    latex_content.append(r"\hline")
                    
                    # Add table rows
                    for row, is_header in table_data:
                        row_str = " & ".join([str(cell) for cell in row])
                        latex_content.append(row_str + r" \\")
                        latex_content.append(r"\hline")
                    
                    # Close table
                    latex_content.append(r"\end{tabular}")
                    latex_content.append(r"\caption{Performance Results}")
                    latex_content.append(r"\end{table}")
                    latex_content.append("")
    
    # Close document
    latex_content.append(r"\end{document}")
    
    # Write LaTeX file
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    
    print(f"Converted {html_file} to {latex_file}")
    return latex_file

def main():
    # Convert HTML to LaTeX
    html_file = "CyberThreat-ML_Research_Paper_Updated.html"
    latex_file = "CyberThreat-ML_Research_Paper.tex"
    
    latex_file = html_to_latex(html_file, latex_file)
    
    # Try to compile LaTeX to PDF
    try:
        os.system(f"pdflatex {latex_file}")
        os.system(f"pdflatex {latex_file}")  # Run twice for references
        
        # Check if PDF was created
        pdf_file = latex_file.replace(".tex", ".pdf")
        if os.path.exists(pdf_file):
            print(f"Successfully created {pdf_file}")
        else:
            print(f"Failed to create PDF. Please check for LaTeX compilation errors.")
    except Exception as e:
        print(f"Error compiling LaTeX: {e}")
        print("Creating LaTeX file only. You can compile it separately to PDF.")

if __name__ == "__main__":
    main()