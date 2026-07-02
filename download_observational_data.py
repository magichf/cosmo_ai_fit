#!/usr/bin/env python
"""下載真實觀測數據（Planck 2018 + DESI DR1）

用法：
    python download_observational_data.py
    python download_observational_data.py --planck-only
    python download_observational_data.py --desi-only
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
import urllib.error
import tarfile
import shutil


class DataDownloader:
    """觀測數據下載器"""
    
    def __init__(self):
        """初始化"""
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("宇宙學觀測數據下載工具")
        print("="*80 + "\n")
    
    def download_planck_2018(self):
        """下載Planck 2018數據"""
        print("[1/2] 下載Planck 2018 CMB數據...")
        print("-" * 80)
        
        planck_dir = self.data_dir / "planck"
        planck_dir.mkdir(exist_ok=True)
        
        print("\n  Planck 2018數據文件：")
        print("  - High-ℓ TT,TE,EE (ℓ = 30-2500)")
        print("  - Low-ℓ TT,EE,TE (ℓ = 2-29)")
        print("  - Foreground likelihood")
        print("\n  數據來源：")
        print("  https://pla.esac.esa.int/pla/")
        
        # 下載URL
        urls = {
            "planck_2018_cl.dat": "https://pla.esac.esa.int/pla-sl/data-footer/COM_PowerSpect_CMB-TT-lowhigh_R3.01.txt",
            "planck_2018_cov_inv.dat": "https://pla.esac.esa.int/pla-sl/data-footer/COM_PowerSpect_CMB-TT-lowhigh_R3.01.cov",
        }
        
        print(f"\n  保存位置: {planck_dir}")
        print(f"\n  ⓘ 手動下載說明：")
        print(f"    1. 訪問 https://pla.esac.esa.int/pla/")
        print(f"    2. 進入 'Explanatory Supplement' → 'Low resolution maps'")
        print(f"    3. 下載 COM_PowerSpect_CMB*.txt")
        print(f"    4. 保存到 {planck_dir}/\n")
    
    def download_desi_dr1(self):
        """下載DESI DR1數據"""
        print("\n[2/2] 下載DESI DR1數據...")
        print("-" * 80)
        
        desi_dir = self.data_dir / "desi"
        desi_dir.mkdir(exist_ok=True)
        
        print("\n  DESI DR1數據文件：")
        print("  - BAO measurements (z = 0.51, 0.71, 0.95)")
        print("  - RSD measurements")
        print("  - P(k) measurements (k = 0.01-0.2 h/Mpc)")
        print("  - Covariance matrices")
        print("\n  數據來源：")
        print("  https://data.desi.lbl.gov/")
        
        print(f"\n  保存位置: {desi_dir}")
        print(f"\n  ⓘ 手動下載說明：")
        print(f"    1. 訪問 https://data.desi.lbl.gov/dr1")
        print(f"    2. 進入 'BAO' 或 'RSD' 部分")
        print(f"    3. 下載BAO + RSD + P(k)文件")
        print(f"    4. 保存到 {desi_dir}/\n")
    
    def generate_mock_files(self):
        """生成模擬數據文件（用於測試）"""
        print("\n生成模擬數據用於測試...\n")
        
        planck_dir = self.data_dir / "planck"
        desi_dir = self.data_dir / "desi"
        planck_dir.mkdir(exist_ok=True)
        desi_dir.mkdir(exist_ok=True)
        
        import numpy as np
        
        # 模擬Planck數據
        n_ell = 2499
        ell = np.arange(2, n_ell + 2)
        Cl = 2700 * (ell / 220) ** (-1.1) / (1 + (ell / 800) ** 2)
        
        planck_data = np.column_stack([ell, Cl])
        np.savetxt(planck_dir / "planck_2018_cl.dat", planck_data)
        
        cov_mock = np.eye(len(ell)) * 0.01
        np.savetxt(planck_dir / "planck_2018_cov_inv.dat", cov_mock)
        
        print(f"  ✓ Planck模擬數據: {len(ell)} 多極")
        
        # 模擬DESI數據
        n_k = 20
        k = np.logspace(-2, -0.7, n_k)
        Pk = 1000 * (k / 0.1) ** 0.96
        
        desi_data = np.column_stack([k, Pk])
        np.savetxt(desi_dir / "desi_dr1_pk.dat", desi_data)
        
        cov_mock_desi = np.eye(len(k)) * 0.05
        np.savetxt(desi_dir / "desi_dr1_cov_inv.dat", np.linalg.inv(cov_mock_desi))
        
        print(f"  ✓ DESI模擬數據: {len(k)} k-bins")
        print(f"\n  所有模擬文件已保存到 {self.data_dir}/\n")
    
    def download_all(self):
        """下載所有數據"""
        self.download_planck_2018()
        self.download_desi_dr1()
        
        print("\n" + "="*80)
        print("數據下載指南")
        print("="*80)
        print("\n選項 A：自動使用模擬數據")
        print(f"  python download_observational_data.py --mock")
        print(f"  → 自動生成測試數據在 {self.data_dir}/\n")
        
        print("選項 B：手動下載真實數據")
        print("  Planck 2018:")
        print("    訪問: https://pla.esac.esa.int/pla")
        print(f"    保存到: {self.data_dir}/planck/")
        print("\n  DESI DR1:")
        print("    訪問: https://data.desi.lbl.gov")
        print(f"    保存到: {self.data_dir}/desi/")
        print("\n選項 C：使用conda下載官方似然")
        print("  conda install -c conda-forge clik planck-likelihood")
        print("\n" + "="*80 + "\n")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="下載觀測數據用於宇宙學擬合"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="生成模擬數據用於測試（不下載真實數據）"
    )
    parser.add_argument(
        "--planck-only",
        action="store_true",
        help="僅下載Planck數據"
    )
    parser.add_argument(
        "--desi-only",
        action="store_true",
        help="僅下載DESI數據"
    )
    
    args = parser.parse_args()
    
    downloader = DataDownloader()
    
    if args.mock:
        downloader.generate_mock_files()
    else:
        downloader.download_all()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
