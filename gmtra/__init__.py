"""
Source code for Global Multihazard Transport Risk Analysis (GMTRA)

Copyright (C) 2019 Elco Koks. All versions released under the GNU Affero General Public License v3.0 license.
"""
import pkg_resources

__author__ = "Elco Koks"
__copyright__ = "Elco Koks"
__license__ = "MIT"

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = 'unknown'
    
__all__ = ['utils','preprocessing','fetch','hazard','exposure','damage','sensitivity','parallel','summary']
