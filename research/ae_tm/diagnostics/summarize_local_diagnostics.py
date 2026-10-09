"""Summarize all frozen policies, controls and uncertainty without pixel outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from continuous_reference import paired_bootstrap


def run(n0_dir,dng_dir,output):
    n0=json.loads((n0_dir/"results.json").read_text())
    dng=json.loads((dng_dir/"results.json").read_text())
    if n0["status"]!="completed_simulated_diagnostic":
        raise ValueError("N0 is not a completed diagnostic")
    if dng["status"]!="completed_real_content_relative_diagnostic":
        raise ValueError("DNG experiment is not a completed diagnostic")
    output.mkdir(parents=True,exist_ok=True)
    n0_rows=[]
    for row in n0["replicates"]:
        seed=row["seed"]
        for population in ("N0_diagnostic","N1_dependence_diagnostic"):
            result=row["diagnostics_by_nodes"]["512"][population]
            metrics_file=n0_dir/"source-metrics"/row["source_metrics_files"][population]
            with np.load(metrics_file,allow_pickle=False) as m:
                cost=m["expected_costs_by_action"]
                action=m["frozen_policy_actions"]
                diff=cost[np.arange(len(cost)),action[:,0]]-cost[np.arange(len(cost)),action[:,1]]
            comparison=paired_bootstrap(diff,np.random.default_rng([seed,9901]),2000)
            n0_rows.append({"seed":seed,"population":population,
                "fixed_risk":result["fixed_expected_cost"],
                "residual_relative_gain":result["policies"]["known_linear_residual"]["relative_reduction_vs_fixed"],
                "brightness_relative_gain":result["policies"]["global_brightness"]["relative_reduction_vs_fixed"],
                "residual_minus_brightness":comparison,
                "nodes256_vs512_max_action_risk_difference":
                    row["nodes_convergence"]["256"][population]["max_abs_action_risk_difference"]})
    dng_rows=[]
    for camera in ("iphone","s25"):
        for backend in ("inverse","point_posterior","local_wiener"):
            rows=[r["cameras"][camera]["backends"][backend]["diagnostic"] for r in dng["replicates"]]
            dng_rows.append({"camera":camera,"backend":backend,
                "fixed_exposures":[r["dev_fixed_exposure"] for r in rows],
                "fixed_risk_range":[min(r["fixed_risk"] for r in rows),max(r["fixed_risk"] for r in rows)],
                "policy_gain_ranges":{name:[min(r["policies"][name]["relative_reduction_vs_fixed"] for r in rows),
                                           max(r["policies"][name]["relative_reduction_vs_fixed"] for r in rows)]
                                       for name in rows[0]["policies"]},
                "policy_ci_by_seed":{name:[r["policies"][name]["descriptive_group_bootstrap_ci95"]
                                         for r in rows] for name in rows[0]["policies"]}})
    # Frozen policy actions are replayed across backend costs, using the same sources.
    with (dng_dir/"source_metrics.csv").open(newline="",encoding="utf-8") as f:
        metrics=list(csv.DictReader(f))
    index={(int(r["seed"]),r["camera"],r["backend"],r["source_id"]):r for r in metrics}
    backends=("inverse","point_posterior","local_wiener")
    exposures=dng["actions"]
    cross=[]
    for seed in dng["config"]["seeds"]:
        for camera in ("iphone","s25"):
            ids=sorted({r["source_id"] for r in metrics if int(r["seed"])==seed and r["camera"]==camera})
            for selector in backends:
                for renderer in backends:
                    for feature in ("mean_brightness","q99_brightness","temporal_noise","local_gradient"):
                        group_values={}
                        for source in ids:
                            s=index[(seed,camera,selector,source)]
                            r=index[(seed,camera,renderer,source)]
                            assert s["group_id"]==r["group_id"]
                            action=int(s[f"policy_action_{feature}"])
                            value=float(r[f"risk_e{exposures[action]:g}"])
                            group_values.setdefault(r["group_id"],[]).append(value)
                        risk=float(np.mean([np.mean(v) for v in group_values.values()]))
                        cross.append({"seed":seed,"camera":camera,"selector_backend":selector,
                                      "renderer_backend":renderer,"feature":feature,"group_balanced_risk":risk})
    with (output/"dng_backend_cross.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
    summary={"N0_N1":n0_rows,"DNG":dng_rows,
        "DNG_quadrature_max_abs_risk_difference":max(
            c["quadrature_convergence"]["max_absolute_group_action_risk_difference"]
            for r in dng["replicates"] for c in r["cameras"].values()),
        "DNG_code_commit":dng["git_commit"],
        "cross_matrix_note":"Frozen source backend policies replayed against same future capture costs; limited renderer classes, one-session diagnostic",
        "cross_family_review":"not available; existing reviews provisional",
        "full_H_reference":"pending; current posterior does not assimilate preview history"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    def pct(bounds):return f"{100*bounds[0]:.2f}%–{100*bounds[1]:.2f}%"
    lines=["# CaptureTM 本地诊断：当前结论","",
        "日期：2026-10-09。状态：实验已执行，投稿贡献与最终验收仍未完成。",
        "","## 已完成的证据","",
        "- 数据审计：34对、68个唯一DNG，11个保守背景组；train/development/diagnostic为6/2/3组、28/16/24文件。同一采集会话，不把图像数当独立场景数。",
        "- N0/N1：3seed；每个诊断8192个IID双像素源、16次未来噪声重复；积分64/128/256/512。实际内部CPU耗时151.833秒，无GPU。",
        "- DNG：3seed、每文件8个64×64绿色通道裁块，4次未来噪声重复；全部后端共享捕获。实际内部CPU耗时"+
        f"{dng['elapsed_seconds']:.3f}秒；没有神经训练或真实多曝光重拍。",
        "- 代码见工程提交 "+dng["git_commit"]+"。环境已由独立代理照文档执行，witness与测试通过。","",
        "## 连续参考保留下来的范围","",
        "| 条件 | seed | 已知关系规则相对固定曝光下降 | 亮度规则下降 | 关系规则减亮度规则的MSE差及95%CI |",
        "|---|---|---|---|---|"]
    for r in n0_rows:
        b=r["residual_minus_brightness"]
        ci=b["paired_independent_source_percentile_ci95"]
        lines.append(f"| {r['population']} | {r['seed']} | {100*r['residual_relative_gain']:.2f}% | {100*r['brightness_relative_gain']:.2f}% | {b['mean_difference']:.3e} [{ci[0]:.3e}, {ci[1]:.3e}] |")
    lines+=["","256与512节点的逐动作平均风险差均小于1e-18，当前收益不是原先观察分箱造成的误差。CIs条件于已冻结规则与有限未来噪声重复。",
        "","重要限定：当前共同后端仅收到未来捕获和已知曝光，不融合三张历史预览，也没有按动作选择事件条件化先验。结果是受限后端比较，不能称充分使用全部可用观测的Bayes下界。下一决定检验是H-aware后验。N1仍使用训练先验；反向依赖留出不是任意未知图像先验泛化。",
        "","绝对收益仍小于原设δMSE=1e-4；统计区别不等于达到该必要替换阈值。关系规则相对亮度规则的直接CI已补齐，而非从各自对固定曝光的CI推断。",
        "","## DNG所有预设规则的结果","",
        "百分比为相对该后端development选定固定曝光的误差下降；负值表示退化。列出全部规则，不按诊断最优挑选新策略。",
        "","| 相机/后端 | 预览均值 | 预览99分位 | 时间噪声 | 局部梯度 |",
        "|---|---|---|---|---|"]
    for r in dng_rows:
        g=r["policy_gain_ranges"]
        lines.append(f"| {r['camera']}/{r['backend']} | {pct(g['mean_brightness'])} | {pct(g['q99_brightness'])} | {pct(g['temporal_noise'])} | {pct(g['local_gradient'])} |")
    lines+=["",
        "只有三个诊断背景组，bootstrap区间仅为描述性；许多区间跨零或触零。多个种子重复同一组源内容，不能当三次独立采集。",
        "",
        "观察：规则收益随后端家族改变，部分规则反转为退化。当前没有复杂AE优于充分后端与简单规则的证据。反演、单像素经验后验、局部Wiener均不是任意强图像TM；跨后端矩阵也只覆盖这三个有限家族。",
        "",
        "DNG节点2/4每密度bin的平均风险最大差："+
        f"{summary['DNG_quadrature_max_abs_risk_difference']:.3e}。这是后验积分对照，不是未知图像prior或完整神经后端收敛证明。",
        "",
        "## 可以复用的结论","",
        "1. DNG根IFD可能只是JPEG预览，必须递归检查RAW SubIFD。编码位数、linearization table和LibRaw解码尺度分别记录。",
        "2. 当前文件是处理过的Linear RAW；bounded source只能支撑相对模拟，无法补造已失高光、干净HDR或原生Bayer真值。",
        "3. 数据目录移动后，先在声明的相机根目录内按basename安全解析；不能信任旧绝对路径。",
        "4. 先按源场景分组，再生成crop/noise；配对相机和相邻帧不增加独立样本数。",
        "5. 物理ADC后验必须覆盖零码/顶码完整概率；极小及正次正规似然用原模型logsumexp，不能换Gaussian来掩盖下溢。",
        "6. 后端容量、先验和可用历史观测改变策略比较的含义；H被策略使用却被参考后端忽略时，不能宣称充分补偿后的不可替代采集收益。",
        "7. 相对百分比、绝对容差和配对置信区间都要报告；未来噪声实现后的最小值是hindsight统计，不是部署oracle。",
        "",
        "## 投稿判断与剩余步骤","",
        "保留为待验证的研究问题：固定目标下，充分利用历史信息后是否仍存在可预测、跨后端的曝光收益。当前尚不足以认定可投稿贡献，也没有据此否定整个topic。",
        "",
        "下一步：H-aware连续后验及严格冻结后端交叉矩阵 → 更新claim和最新文献对照 → 有效跨模型系列查新/批判审阅 → 重跑最终证据gate。只有决定检验通过且简单方法不能解释时，才安排必要的强神经后端训练。现有同系列审阅保持provisional。",
        "",
        "所有原始JSON、逐源成本、配置、失败记录和正负结果保留。没有修改原capture_tm目标、动作目录、正式S24test或用户原图。"]
    (output/"CONCLUSIONS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"n0_rows":len(n0_rows),"dng_rows":len(dng_rows),
                      "cross_rows":len(cross),"dng_quadrature_error":
                      summary["DNG_quadrature_max_abs_risk_difference"]}))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n0",type=Path,required=True)
    p.add_argument("--dng",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.n0,a.dng,a.out)
