# Sphinx documentation build configuration file
import os
import sys

# Add src to path so autodoc can find the modules
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information ------------------------------------------------
project = 'PyTorch Mastery Hub'
copyright = '2026, PyTorch Mastery Hub Contributors'
author = 'PyTorch Mastery Hub Contributors'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'myst_parser',
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for NumPy docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch', 'torchvision', 'torchaudio',
    'numpy', 'scipy', 'pandas', 'sklearn',
    'matplotlib', 'seaborn', 'plotly',
    'PIL', 'cv2', 'albumentations',
    'transformers', 'datasets', 'optuna', 'wandb',
    'nltk', 'spacy', 'tokenizers',
    'librosa', 'soundfile',
    'lightning',
]

# Intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# MyST parser settings (for Markdown)
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'tasklist',
]

# nbsphinx settings
nbsphinx_execute = 'never'  # Never execute notebooks during docs build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# Todo extension
todo_include_todos = True

# Template paths
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Source suffix
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    "display_github": True,
    "github_user": "SatvikPraveen",
    "github_repo": "pytorch-mastery-hub",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for LaTeX output ------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'figure_align': 'htbp',
}

# -- Extension configuration -------------------------------------------
# If true, show TODO items in the output
todo_include_todos = True
