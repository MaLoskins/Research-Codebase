\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cshonours}[2025/05/14 Substitute class for CS Honours dissertation]

% Load the report class as the base
\LoadClass[12pt,a4paper]{report}

% Define basic formatting
\RequirePackage{geometry}
\geometry{margin=1in}

% Support for title elements
\def\keywords#1{\gdef\@keywords{#1}}
\def\@keywords{\@latex@warning@no@line{No \noexpand\keywords given}}

\def\categories#1{\gdef\@categories{#1}}
\def\@categories{\@latex@warning@no@line{No \noexpand\categories given}}

% Redefine maketitle to include keywords and categories
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \begin{center}
      \vspace*{2em}
      {\Large\bfseries\@title\par}
      \vspace{1.5em}
      {\large\@author\par}
      \vspace{1em}
      \@date
      \vspace{2em}

      \textbf{Keywords:} \@keywords\par
      \vspace{0.5em}
      \textbf{Categories:} \@categories\par
    \end{center}
  \end{titlepage}
}

% Load commonly used packages
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{hyperref}

% Set up bibliography
\bibliographystyle{acm}