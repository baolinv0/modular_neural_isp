# Tone Mapping 方案与伪 GT 探针归档

归档日期：2026-10-17。类型：既有讨论与附件归档；未运行实验、未修改模型代码。

## 阅读顺序

1. [五种候选方案完整报告](TM_Five_Candidate_Paradigms_2026-10-17.md)：动机、公式、实现边界、12项参考资料及判别实验。
2. [原始项目定义](sources/Definition.txt)：iPhone训练 baseline → S24迁移 → 同场景非对齐参考生成候选伪GT。
3. [单场景合同](sources/CONTRACT.md)、[实施计划](sources/PLAN.md)、[验收清单](sources/IMPLEMENTATION_CHECKLIST.md)。这些只约束伪GT探针，不是五种架构的统一合同。
4. [通用机制知识](https://github.com/baolinv0/CameraPaper/blob/main/knowledge/hdr-tone-mapping/local-tone-mapping-paradigms.md)。

## 原始文件清单

所有文件保持字节内容原样；附件放入 sources/，避免覆盖仓库已有同名文件。两份README内容相同，均保留以完整对应上传附件。

| 原始文件 | 归档位置 | SHA-256 |
|---|---|---|
| TM_Five_Candidate_Paradigms_2026-10-17.md | [文件](TM_Five_Candidate_Paradigms_2026-10-17.md) | `d33fa1cea745635c9eff569ed3fb094d0a1cfada7aa2cac389d707d116de4252` |
| PLAN.md | [文件](sources/PLAN.md) | `929e373c413cba0d16cfd8ffbaa4fd481158968f8320b187912f327cddf74936` |
| CONTRACT.md | [文件](sources/CONTRACT.md) | `232d88b269544a67122692d057a8f06927a236c0cfb9f946bc4932a1145648e4` |
| README(2).md | [文件](sources/README(2).md) | `aaa2eeb05cd6f6313ebd13b52ac68f603c0c59248567829e295c4a190dcb760d` |
| IMPLEMENTATION_CHECKLIST.md | [文件](sources/IMPLEMENTATION_CHECKLIST.md) | `b432fcda2db200fa78eb509732a46b51b05305d9ddf61df17fbe31e39e0b3f98` |
| README.md.txt | [文件](sources/README.md.txt) | `aaa2eeb05cd6f6313ebd13b52ac68f603c0c59248567829e295c4a190dcb760d` |
| Definition.txt | [文件](sources/Definition.txt) | `fc313781fac0219ef9b5bb1fb9a242ad560c03d97dcd3fa24da07526f6bbca79` |

## 状态与边界

五种候选尚待统一实验筛选；这里的A–E是2026-10-17讨论定义，不默认与历史分支的同名编号一一对应。未更新历史训练配置或合并架构代码。附件模型标识与能力为原合同记录，本次没有重新验证API或模型可用性。非对齐参考不能作为像素级GT；单场景通过不代表训练收益或泛化。完整报告中“本次核验”指原报告撰写轮次；这次仅归档。
