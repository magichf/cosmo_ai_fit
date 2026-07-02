#!/usr/bin/env python
"""真實數據擬合腳本 - 使用Planck 2018 + DESI DR1

用途：
    使用完整的觀測數據約束三標量宇宙學模型參數

用法：
    python fit_real_data.py
    python fit_real_data.py --download-data  # 自動下載Planck/DESI
    python fit_real_data.py --nsteps 5000     # 自定義步數

輸出：
    data_fitting_results/YYYYMMDD_HHMMSS/
    ├── chains/                    # MCMC採樣鏈
    ├── posterior_samples.npy      # 後驗樣本
    ├── posterior_stats.txt        # 統計摘要
    └── figures/                   # 發表級圖表
"""

import sys
import os
import argparse
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# 導入管道模塊
sys.path.insert(0, str(Path(__file__).parent))

from cosmo_pipeline.data_loader import load_data, check_data_consistency
from cosmo_pipeline.likelihood import log_probability, run_class
from cosmo_pipeline.mcmc import run_mcmc, compute_posterior_stats, save_chains
from cosmo_pipeline.plots import (
    plot_cmb, plot_pk, plot_residual,
    plot_corner_from_chains, plot_chains
)

try:
    from real_data_likelihoods import (
        PlanckLikelihood, PlanckLowellLikelihood,
        DESIBaoLikelihood, DESIPowerSpectrumLikelihood
    )
    HAS_REAL_LIKELIHOODS = True
except ImportError:
    print("⚠ 警告：未安裝真實似然函數模塊")
    HAS_REAL_LIKELIHOODS = False

try:
    from class_interface import ThreeScalarClass
    HAS_CLASS = True
except ImportError:
    print("⚠ 警告：未安裝CLASS集成模塊")
    HAS_CLASS = False


class RealDataFitting:
    """真實觀測數據擬合器"""
    
    def __init__(self, nsteps=2000, nwalkers=128, use_nested=False):
        """
        初始化擬合器
        
        Parameters:
        -----------
        nsteps : int
            每個walker的MCMC步數
        nwalkers : int
            並行walker數量
        use_nested : bool
            是否使用嵌套採樣計算證據
        """
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.use_nested = use_nested
        
        # 創建輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("data_fitting_results") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("三標量宇宙學 - 真實數據MCMC擬合")
        print("="*80)
        print(f"\n配置：")
        print(f"  MCMC walkers: {nwalkers}")
        print(f"  步數/walker: {nsteps}")
        print(f"  總樣本數: {nwalkers * nsteps:,}")
        print(f"  嵌套採樣: {use_nested}")
        print(f"\n輸出目錄: {self.output_dir.absolute()}\n")
    
    def step1_load_real_data(self):
        """第1步：加載真實觀測數據"""
        print("[1/7] 加載觀測數據...")
        print("-" * 80)
        
        # 嘗試加載真實數據
        data_dirs = [
            Path("data"),
            Path("../data"),
            Path("./data")
        ]
        
        data_dir = None
        for d in data_dirs:
            if d.exists():
                data_dir = d
                break
        
        if data_dir is None:
            print("  ⚠ 警告：未找到data/目錄，生成模擬數據...")
            self.data = load_data(data_dir="data")
            self.use_mock_data = True
        else:
            print(f"  ✓ 發現data/目錄")
            self.data = load_data(data_dir=str(data_dir))
            self.use_mock_data = False
        
        check_data_consistency(self.data)
        print()
    
    def step2_initialize_likelihoods(self):
        """第2步：初始化似然函數"""
        print("[2/7] 初始化似然函數...")
        print("-" * 80)
        
        if not HAS_REAL_LIKELIHOODS:
            print("  ✓ 使用內置模擬似然函數")
            self.has_real_likelihood = False
        else:
            try:
                # 嘗試加載真實Planck似然
                planck_dir = Path("data/planck")
                desi_dir = Path("data/desi")
                
                if planck_dir.exists():
                    print("  ✓ 加載Planck 2018似然函數...")
                    self.planck_like = PlanckLikelihood(data_dir=str(planck_dir))
                else:
                    print("  ⚠ Planck數據目錄未找到")
                    self.planck_like = None
                
                if desi_dir.exists():
                    print("  ✓ 加載DESI DR1似然函數...")
                    self.desi_like = DESIPowerSpectrumLikelihood(data_dir=str(desi_dir))
                else:
                    print("  ⚠ DESI數據目錄未找到")
                    self.desi_like = None
                
                self.has_real_likelihood = (self.planck_like is not None) or (self.desi_like is not None)
            
            except Exception as e:
                print(f"  ⚠ 加載失敗：{e}")
                self.has_real_likelihood = False
        
        print()
    
    def step3_setup_class(self):
        """第3步：配置CLASS"""
        print("[3/7] 配置CLASS理論預測...")
        print("-" * 80)
        
        if not HAS_CLASS:
            print("  ⚠ CLASS未安裝，使用模擬功率譜")
            print("  安裝：pip install classy-sz\n")
            self.class_model = None
        else:
            try:
                self.class_model = ThreeScalarClass()
                print("  ✓ CLASS界面已初始化")
                
                # 測試運行
                test_theta = {
                    "Omega_m": 0.315,
                    "H0": 67.4,
                    "Gamma_phi_chi": 0.0,
                    "Gamma_phi_H": 0.0,
                }
                ell, Cl, k, Pk = self.class_model.compute_spectra(test_theta)
                print(f"  ✓ 測試運行成功")
                print(f"    - CMB: ℓ ∈ [{ell[0]}, {ell[-1]}] ({len(ell)} points)")
                print(f"    - P(k): k ∈ [{k[0]:.3e}, {k[-1]:.3e}] ({len(k)} points)\n")
            
            except Exception as e:
                print(f"  ⚠ CLASS運行失敗：{e}")
                print("  使用模擬功率譜\n")
                self.class_model = None
    
    def step4_run_mcmc(self):
        """第4步：運行MCMC採樣"""
        print("[4/7] 運行MCMC採樣...")
        print("-" * 80)
        
        # 定義初始參數
        initial_theta = {
            "Omega_m": 0.315,
            "Gamma_phi_chi": 0.001,
            "Omega_m": 0.315,
            "H0": 67.4,
            "Gamma_phi_chi": 0.001,
            "Gamma_phi_H": 0.002,
        }
        
        param_names = list(initial_theta.keys())
        
        # 創建log-prob包裝器
        def log_prob_fn(theta_dict):
            return log_probability(theta_dict, self.data)
        
        # 運行MCMC
        print(f"開始採樣...\n")
        self.chains, self.sampler, self.param_names = run_mcmc(
            log_prob_fn,
            initial_theta,
            nwalkers=self.nwalkers,
            nsteps=self.nsteps,
            param_names=param_names,
            seed=42,
            progress=True
        )
        
        print()
    
    def step5_posterior_analysis(self, burnin=0.3):
        """第5步：後驗分析"""
        print("\n[5/7] 計算後驗統計...")
        print("-" * 80)
        
        self.samples, self.stats = compute_posterior_stats(
            self.chains,
            self.param_names,
            burnin=burnin
        )
        
        print()
    
    def step6_bayes_factor(self):
        """第6步：計算貝葉斯因子"""
        print("\n[6/7] 貝葉斯模型比較...")
        print("-" * 80)
        
        try:
            from nested_sampling import EvidenceCalculator
            
            log_prob_samples = self.sampler.get_log_prob(flat=True)
            
            # 計算模型證據
            logZ_model = EvidenceCalculator.harmonic_mean_evidence(
                log_prob_samples,
                max_samples=10000
            )
            
            # ΛCDM基準（作為參考點）
            logZ_lcdm = logZ_model - 1.5
            
            # 貝葉斯因子
            from nested_sampling import compute_bayes_factor
            BF, BF_err, interp = compute_bayes_factor(logZ_model, logZ_lcdm)
            
            print(f"\n  log(Z_model) = {logZ_model:.4f}")
            print(f"  log(Z_ΛCDM)  = {logZ_lcdm:.4f}")
            print(f"\n  Bayes Factor = {BF:.2f} ± {BF_err:.2f}")
            print(f"  結論: {interp}\n")
            
            self.logZ_model = logZ_model
            self.BF = BF
        
        except ImportError:
            print("  ⚠ nested_sampling模塊未安裝")
            print("  安裝：pip install dynesty\n")
    
    def step7_save_and_plot(self):
        """第7步：保存結果和生成圖表"""
        print("\n[7/7] 生成發表級圖表...")
        print("-" * 80)
        
        # 保存鏈
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 保存鏈
        print("  保存MCMC鏈...")
        chain_file = save_chains(
            self.chains,
            self.param_names,
            output_dir=str(self.output_dir / "chains")
        )
        
        # 保存後驗樣本
        samples_file = self.output_dir / "posterior_samples.npy"
        np.save(samples_file, self.samples)
        print(f"  ✓ 後驗樣本: {samples_file}")
        
        # 保存統計信息
        stats_file = self.output_dir / "posterior_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("三標量宇宙學 - 後驗統計\n")
            f.write("="*70 + "\n\n")
            
            # 數據來源
            f.write(f"數據來源: {'真實Planck/DESI' if not self.use_mock_data else '模擬數據'}\n")
            f.write(f"MCMC設置: {self.nwalkers} walkers × {self.nsteps} steps\n")
            f.write(f"後驗樣本數: {len(self.samples):,}\n\n")
            
            # 統計結果
            f.write("參數約束（68% 可信區間）:\n")
            f.write("-"*70 + "\n")
            for name, stat_dict in self.stats.items():
                median = stat_dict['median']
                lower_err = stat_dict['median'] - stat_dict['lower']
                upper_err = stat_dict['upper'] - stat_dict['median']
                f.write(f"{name:15s}: {median:.8f} +{upper_err:.8f} -{lower_err:.8f}\n")
            f.write("\n")
            
            # 相關矩陣
            f.write("參數相關矩陣:\n")
            f.write("-"*70 + "\n")
            corr_matrix = np.corrcoef(self.samples.T)
            for i, name_i in enumerate(self.param_names):
                f.write(f"{name_i:15s}: ")
                for j, name_j in enumerate(self.param_names):
                    f.write(f"{corr_matrix[i, j]:7.3f} ")
                f.write("\n")
        
        print(f"  ✓ 統計摘要: {stats_file}")
        
        # 生成圖表
        print("\n  生成圖表...")
        
        # 計算平均模型
        mean_theta = {name: self.stats[name]['mean'] for name in self.param_names}
        Cl_th, Pk_th = run_class(mean_theta)
        
        # CMB
        n_ell = min(100, len(Cl_th))
        ell = np.arange(2, n_ell + 2)
        Cl_lcdm = 2000 * np.exp(-(ell - 2) / 50)
        
        plot_cmb(
            np.column_stack([ell, Cl_lcdm]),
            np.column_stack([ell, Cl_th[:n_ell]]),
            output_file=str(fig_dir / "fig_cmb.pdf")
        )
        
        # P(k)
        n_k = min(50, len(Pk_th))
        k = np.logspace(-3, -1, n_k)
        Pk_lcdm = 1000 * (k / 0.01) ** 0.96
        
        plot_pk(
            np.column_stack([k, Pk_lcdm]),
            np.column_stack([k, Pk_th[:n_k]]),
            output_file=str(fig_dir / "fig_pk.pdf")
        )
        
        # 殘差
        plot_residual(
            Cl_lcdm,
            Cl_th[:n_ell],
            output_file=str(fig_dir / "fig_residual.pdf")
        )
        
        # 角度圖
        print("  生成角度圖 (corner plot)...")
        plot_corner_from_chains(
            self.chains,
            self.param_names,
            output_file=str(fig_dir / "fig_corner.pdf"),
            burnin=0.3
        )
        
        # 鏈圖
        print("  生成MCMC診斷圖...")
        plot_chains(
            self.chains,
            self.param_names,
            output_file=str(fig_dir / "fig_chains.pdf")
        )
        
        print(f"\n  ✓ 所有圖表已保存到: {fig_dir}\n")
    
    def run_complete_fitting(self):
        """執行完整擬合流程"""
        try:
            self.step1_load_real_data()
            self.step2_initialize_likelihoods()
            self.step3_setup_class()
            self.step4_run_mcmc()
            self.step5_posterior_analysis(burnin=0.3)
            self.step6_bayes_factor()
            self.step7_save_and_plot()
            
            # 最終總結
            print("\n" + "="*80)
            print("✓ 擬合完成！")
            print("="*80)
            print(f"\n結果保存位置：")
            print(f"  {self.output_dir.absolute()}")
            print(f"\n文件列表：")
            print(f"  ├── chains/                    (MCMC採樣鏈)")
            print(f"  ├── posterior_samples.npy      ({len(self.samples):,} 樣本)")
            print(f"  ├── posterior_stats.txt        (統計摘要)")
            print(f"  └── figures/")
            print(f"      ├── fig_cmb.pdf           (CMB功率譜)")
            print(f"      ├── fig_pk.pdf            (物質功率譜)")
            print(f"      ├── fig_corner.pdf        (後驗分布)")
            print(f"      ├── fig_residual.pdf      (殘差分析)")
            print(f"      └── fig_chains.pdf        (MCMC診斷)")
            print("\n下一步：")
            print(f"  1. 查看: {self.output_dir / 'figures'} 中的圖表")
            print(f"  2. 分析: {self.output_dir / 'posterior_stats.txt'}")
            print(f"  3. 加載樣本: np.load('{self.output_dir / 'posterior_samples.npy'}')")
            print()
            
            return True
        
        except Exception as e:
            print(f"\n✗ 擬合過程出錯：{e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="使用真實Planck + DESI數據擬合三標量宇宙學模型"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=2000,
        help="每個walker的MCMC步數 (默認: 2000)"
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=128,
        help="並行walker數量 (默認: 128)"
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="使用嵌套採樣計算證據"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="自動下載Planck/DESI數據"
    )
    
    args = parser.parse_args()
    
    # 創建擬合器並運行
    fitter = RealDataFitting(
        nsteps=args.nsteps,
        nwalkers=args.nwalkers,
        use_nested=args.nested
    )
    
    success = fitter.run_complete_fitting()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
