# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:43:19 2023

@author: https://pwsiegel.github.io/tech/gitroot/
"""

import git
from pathlib import Path

def get_project_root():
    return str(Path(git.Repo('.', search_parent_directories=True).working_tree_dir))