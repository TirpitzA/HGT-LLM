"""
Dynamic Instance Graph (DIG) Construction & Schema-Based Prompting
Generalized Version Driven by YAML Config
"""

import numpy as np

class BearingDynamicInstanceGraph:
    """
    Constructs a Dynamic Instance Graph (DIG) and narrative rules 
    for the Bearing Diagnosis Model mapped by config.
    """
    def __init__(self, physics_adjacency: np.ndarray, config: dict):
        self.adj_matrix = physics_adjacency
        self.config = config
        
        # Convert config dictionaries to the required formats
        comp_dict = self.config.get('components', {})
        self.component_names = [comp_dict[k] for k in sorted(comp_dict.keys())]
        self.labels = self.config.get('labels', {})
        self.severities = self.config.get('severities', {})
        self.fault_mapping = self.config.get('fault_mapping', {})
        self.reasoning_chains = self.config.get('reasoning_chains', {})

    def process_sample(self, 
                       prediction: int, 
                       confidence: float,
                       edge_attention: np.ndarray, 
                       slice_attention: np.ndarray,
                       lang: str = 'zh'):
        
        critical_slice_idx = int(np.argmax(slice_attention))
        time_importance = float(slice_attention[critical_slice_idx])

        components_status = []
        
        fault_type = self.labels.get(prediction, "Unknown")
        severity_zh = self.severities.get(prediction, "")

        # Node status inference
        fault_keywords = self.fault_mapping.get(prediction, [])
        for name in self.component_names:
            state = "Healthy"
            if prediction != 0:
                is_fault_node = any(kw in name for kw in fault_keywords)
                state = "Faulty" if is_fault_node else "Warning"
            
            components_status.append({
                "name": name,
                "state": state
            })

        explanation = self._generate_report(
            prediction, fault_type, severity_zh, confidence, 
            critical_slice_idx, time_importance,
            components_status, lang
        )
        
        return {
            "prediction": int(prediction),
            "fault_type": fault_type,
            "confidence": float(confidence),
            "critical_time_slice": int(critical_slice_idx),
            "components": components_status,
            "explanation": explanation
        }
        
    def _generate_report(self, pred, fault_type, severity, conf, time_idx, time_score, components, lang):
        if lang == 'zh':
            report = []
            report.append("============ 轴承系统动态实例图(DIG)诊断报告 ============")
            report.append(f"预测状态: {fault_type} (置信度: {conf:.1f}%)")
            report.append("-" * 50)
            
            report.append("1. 时序注意力分析:")
            report.append(f"   模型主要关注时间切片 [{time_idx}] (重要性得分: {time_score:.2f})。这暗示了在此窗口内观测到显著的机械振荡突变。")
            
            report.append("2. 物理拓扑组件状态映射:")
            for c in components:
                state_en = c['state']
                if state_en == "Faulty":
                    report.append(f"   [高活跃] {c['name']} 节点特征空间内观测到剧烈的宽频冲击与异常能量聚集。")
                elif state_en == "Warning":
                    report.append(f"   [次活跃] {c['name']} 节点捕获到受力学拓扑传导引起的受迫振动调制响应。")
                else:
                    report.append(f"   [基线] {c['name']} 节点的频域包络与能量特征维持在设备运转的常规基线范围内。")
                    
            report.append("3. 动态知识图谱传播逻辑:")
            chain = self.reasoning_chains.get(pred, "")
            if "{severity}" in chain:
                chain = chain.format(severity=severity)
            
            # Use chain if available, else omit
            if chain.strip():
                report.append(f"   {chain}")
                 
            return "\n".join(report)
        else:
            return ""
