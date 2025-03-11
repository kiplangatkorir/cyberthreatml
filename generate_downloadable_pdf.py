#!/usr/bin/env python3
"""
Generate a self-contained HTML file for CyberThreat-ML research paper
that can be easily downloaded and converted to PDF.
"""

import os
import base64

def create_self_contained_html(input_html_file, output_html_file):
    """
    Create a self-contained HTML file that embeds MathJax for offline use.
    This makes it easy to download and convert to PDF locally.
    
    Args:
        input_html_file (str): Path to input HTML file
        output_html_file (str): Path to output self-contained HTML file
    """
    # Read the input HTML file
    with open(input_html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Replace the MathJax CDN script with a local version that works offline
    mathjax_cdn = '<script type="text/javascript" id="MathJax-script" async\n    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">\n  </script>'
    
    # Create a simple MathJax configuration that works offline
    offline_mathjax = '''<script type="text/javascript">
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }
    };
    
    // Create a simple polyfill for MathJax to format equations
    document.addEventListener('DOMContentLoaded', function() {
      const equations = document.querySelectorAll('.equation');
      equations.forEach(eq => {
        const formula = eq.textContent.trim();
        // Add some basic styling to equations
        eq.innerHTML = `<div style="font-style: italic; font-size: 1.1em; text-align: center; margin: 1em 0;">${formula}</div>`;
      });
    });
  </script>'''
    
    # Replace the MathJax CDN with our offline version
    html_content = html_content.replace(mathjax_cdn, offline_mathjax)
    
    # Add a download button to the HTML
    download_button = '''
  <div style="text-align: center; margin: 2em 0;">
    <button id="downloadPDF" style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
      Download as PDF
    </button>
    <div style="margin-top: 10px; font-size: 14px; color: #666;">
      <p>To download as PDF:</p>
      <ol style="text-align: left; display: inline-block;">
        <li>Click the button above</li>
        <li>Use your browser's print function (CTRL+P or CMD+P)</li>
        <li>Select "Save as PDF" as the destination</li>
        <li>Click "Save" or "Print"</li>
      </ol>
    </div>
  </div>
  <script>
    document.getElementById('downloadPDF').addEventListener('click', function() {
      window.print();
    });
  </script>
'''
    
    # Insert the download button before the closing body tag
    html_content = html_content.replace('</body>', download_button + '</body>')
    
    # Create additional print-specific CSS
    print_css = '''
    @media print {
      @page {
        size: A4;
        margin: 1cm;
      }
      body {
        font-size: 12pt;
        line-height: 1.5;
      }
      h1 {
        font-size: 18pt;
      }
      h2 {
        font-size: 16pt;
      }
      h3 {
        font-size: 14pt;
      }
      .work-in-progress {
        color: black !important;
      }
      #downloadPDF, #downloadPDF + div {
        display: none;
      }
      pre, code {
        font-size: 10pt;
        white-space: pre-wrap;
        page-break-inside: avoid;
      }
      table {
        page-break-inside: avoid;
      }
      h2, h3 {
        page-break-after: avoid;
      }
      h2 + p, h3 + p {
        page-break-before: avoid;
      }
    }
    '''
    
    # Add the print CSS to the style section
    style_end = '</style>'
    html_content = html_content.replace(style_end, print_css + style_end)
    
    # Write the modified content to the output file
    with open(output_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created self-contained HTML file: {output_html_file}")
    print(f"This file can be downloaded and easily converted to PDF using your browser's print function.")

def main():
    input_html = "CyberThreat-ML_Research_Paper_Updated.html"
    output_html = "CyberThreat-ML_Research_Paper_Downloadable.html"
    
    if not os.path.exists(input_html):
        print(f"Error: Input file {input_html} not found.")
        return
    
    create_self_contained_html(input_html, output_html)

if __name__ == "__main__":
    main()