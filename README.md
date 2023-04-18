# **Alpaca-RLHF-PyTorch**

---
## **Table of Contents**
- [**Alpaca-RLHF-PyTorch**](#alpaca-rlhf-pytorch)
  - [**Table of Contents**](#table-of-contents)
  - [**Environment Setup**](#environment-setup)
  - [**Run**](#run)
    - [**Supervised Finetune**](#supervised-finetune)
    - [**Merge PEFT adapter into Model**](#merge-peft-adapter-into-model)
  - [**Topics**](#topics)
  - [**Reference**](#reference)
---

## **Environment Setup**
```
穷人卡：2080Ti 12G
torch2.0.0
cuda11.8
```

---

- [x] SFT: Supervised Finetune
- [x] Merge Adapter into Model
- [ ] RLHF
  - [x] train reward model
  - [ ] tuning with RL

## **Run**
---
### **Supervised Finetune**

```
python supervised_finetune.py
```


### **Merge PEFT adapter into Model**

```
python merge_peft_adapter.py
```

---

## **Topics**
1. PEFT的版本，目前从git上安装的是 0.3.0.dev0 版本，在merge_peft_adapter的时候有问题，需要切换到peft==0.2.0 (0.3.0.dev0 没有 _get_submodules()这个函数)
2. 

## **Reference**
utils & templates 来自 [alpaca-lora](https://github.com/tloen/
alpaca-lora) 。

requirements 主要是按照 [alpaca-lora](https://github.com/tloen/
alpaca-lora) 来配环境。
* [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
* [https://github.com/lvwerra/trl](https://github.com/lvwerra/trl)
* [https://github.com/jasonvanf/llama-trl](https://github.com/jasonvanf/llama-trl)