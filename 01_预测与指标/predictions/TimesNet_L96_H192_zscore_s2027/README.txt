依次：填写真实运行记录 → 放 validation_pred.npy 与 validation_true.npy → score-validation → seal → 测试推理 → 放 pred.npy 与 true.npy → evaluate。
真值与预测使用元数据声明的同一 prediction_space；起报文件必须来自实际导出顺序核对，不能只按长度假定一致。
