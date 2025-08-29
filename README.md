# X-bar-NLG
This project is an advanced natural language generation system that combines Chomsky's X-bar Theory with a hierarchical, three-level neural network architecture. It generates coherent discourse by first creating a syntactic blueprint (a parse tree) and then expanding it, rather than just predicting word sequences.

# X-bar 理論驅動的篇章生成系統

這是一個基於**Noam Chomsky 的 X-bar 理論**所設計的自然語言生成（NLG）專案。它透過一個創新的三層級神經網路框架，模擬人類語言的結構化過程，從而產生符合語法並具備連貫性的篇章。

與傳統的語言模型（如 GPT）直接預測下一個詞彙不同，本專案的核心思想是：**先建立句法結構，再填充具體內容**。這種方法確保了生成句子的語法正確性與結構合理性。

## 核心架構

本系統由三個層級的模型協同運作：

1.  **句法樹擴展模型 (Level 1):**

      * **功能:** 負責生成單一且符合文法規則的句子。
      * **技術:** 採用 **TreeLSTM** 來處理句法樹的層級資訊，並訓練模型學習一系列「新增節點」、「填充詞彙」的動作序列。
      * **理論基礎:** 嚴格遵循 X-bar 理論的句法規則，確保每個生成的句子都具備完整的結構。

2.  **主題接龍模型 (Level 2):**

      * **功能:** 確保前後句子之間的主題連貫性。
      * **技術:** 透過一個簡單的 **Seq2Seq 模型**，根據前一個句子的「種子子句」（如主詞-動詞-受詞），生成與其相關的新句子。這有助於維持篇章的語義流暢度。

3.  **篇章終止模型 (Level 3):**

      * **功能:** 判斷整個篇章何時應該結束，避免無限生成。
      * **技術:** 是一個基於 LSTM 的二元分類器，會評估當前生成的句子是否適合作為結尾。

## 資料準備與訓練流程

1.  **資料解析:** 使用強大的 `Stanza` NLP 函式庫，對 CNN/DailyMail 語料庫進行句法分析，將每個句子轉換為其句法樹結構。
2.  **動作序列化:** 將每個句法樹解構為一系列可供模型學習的「擴展動作」。
3.  **模型訓練:** 分別訓練三個層級的模型，使其能夠學習各自的生成任務。
4.  **最終生成:** 整合三個模型，實現從主題到篇章的完整生成過程。
