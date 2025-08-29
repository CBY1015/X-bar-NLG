# ============================================================================
# X-bar 理論生成系統 - 最終完整版
# 基於 Chomsky 句法理論的三層級篇章生成框架
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import stanza
from collections import defaultdict
import re
import datasets
import os
import pickle
import random

print("--- X-bar 理論生成系統 - 最終完整版 ---")

# ============================================================================
# 1. 資料結構定義
# ============================================================================
class TreeNode:
    def __init__(self, label, parent=None):
        self.label = label
        self.children = []
        self.parent = parent
        self.id = id(self)

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def copy(self):
        """深度複製節點和所有子節點"""
        new_node = TreeNode(self.label)
        for child in self.children:
            new_node.add_child(child.copy())
        return new_node

    def __repr__(self):
        return f"Node(label='{self.label}', children={len(self.children)})"

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2id = {"<UNK>": 0, "<PAD>": 1, "<START>": 2, "<END>": 3}
        self.id2word = {0: "<UNK>", 1: "<PAD>", 2: "<START>", 3: "<END>"}
        self.word_count = defaultdict(int)
        self.n_words = 4

    def add_item(self, item):
        if item not in self.word2id:
            self.word2id[item] = self.n_words
            self.id2word[self.n_words] = item
            self.n_words += 1
        self.word_count[item] += 1

    def build_from_sequences(self, sequences):
        for seq in sequences:
            for item in seq:
                self.add_item(item)
        print(f"建立 {self.name} 詞彙表，共 {self.n_words} 個項目")

    def save(self, path):
        """儲存詞彙表"""
        with open(path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'word2id': self.word2id,
                'id2word': self.id2word,
                'n_words': self.n_words
            }, f)

    @classmethod
    def load(cls, path):
        """載入詞彙表"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(data['name'])
        vocab.word2id = data['word2id']
        vocab.id2word = data['id2word']
        vocab.n_words = data['n_words']
        return vocab

    def __len__(self):
        return self.n_words

# ============================================================================
# 2. 資料工程函式
# ============================================================================
def extract_seed_clause(tree):
    """從句法樹提取種子子句（主詞-動詞-受詞）"""
    if tree.label != 'ROOT':
        s_node = tree
    else:
        s_node = tree.children[0] if tree.children else None

    if not s_node or s_node.label != 'S':
        return "UNKNOWN"

    subject_np = None
    predicate_vp = None

    for child in s_node.children:
        if child.label == 'NP' and not subject_np:
            subject_np = child
        elif child.label == 'VP' and not predicate_vp:
            predicate_vp = child

    if not subject_np or not predicate_vp:
        return "UNKNOWN"

    def get_head(node):
        if not node.is_preterminal():
            for child in reversed(node.children):
                if child.label.startswith(('NP', 'NN', 'VB')):
                    return get_head(child)
            return get_head(node.children[-1]) if node.children else ""
        else:
            return node.children[0].label

    subject = get_head(subject_np)
    verb, direct_object = "", ""
    
    vp_children = predicate_vp.children
    for i, child in enumerate(vp_children):
        if child.label.startswith('VB'):
            verb = get_head(child)
            if i + 1 < len(vp_children) and vp_children[i+1].label == 'NP':
                direct_object = get_head(vp_children[i+1])
                break

    parts = [part for part in [subject, verb, direct_object] if part]
    return " ".join(parts) if parts else "UNKNOWN"

def generate_expansion_actions(tree):
    """生成擴展動作序列"""
    actions = []
    
    def add_parent_pointers(node, parent=None):
        node.parent = parent
        for child in node.children:
            add_parent_pointers(child, node)
    
    def traverse(node):
        parent_id = id(node.parent) if node.parent else "ROOT"
        actions.append(f"ADD(parent_id={parent_id},new_label={node.label},new_id={id(node)})")
        
        if node.is_preterminal():
            word = node.children[0].label
            actions.append(f"FILL(parent_id={id(node)},content='{word}')")
        
        for child in node.children:
            if not child.is_leaf():
                traverse(child)
        
        actions.append(f"STOP(node_id={id(node)})")

    if tree.children:
        root_node = tree.children[0]
        add_parent_pointers(root_node)
        traverse(root_node)
    
    return actions

# ============================================================================
# 3. 模型架構
# ============================================================================
class TreeLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.W_iou = nn.Linear(input_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, node, node_vocab):
        children_states = [self.forward(child, node_vocab) for child in node.children]
        node_idx = torch.tensor([node_vocab.word2id.get(node.label, 0)])
        x = self.embedding(node_idx)
        
        if not children_states:
            h_sum = torch.zeros(1, self.hidden_dim)
        else:
            children_h = torch.cat([h for h, c in children_states], dim=0)
            h_sum = torch.sum(children_h, dim=0, keepdim=True)
        
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c_sum = torch.zeros(1, self.hidden_dim)
        if children_states:
            children_c = torch.cat([c for h, c in children_states], dim=0)
            children_h = torch.cat([h for h, c in children_states], dim=0)
            f_k = torch.sigmoid(self.W_f(x).repeat(len(children_states), 1) + self.U_f(children_h))
            c_sum = torch.sum(f_k * children_c, dim=0, keepdim=True)

        c_new = i * u + c_sum
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class TreeExpansionModel(nn.Module):
    """層級一：句法樹擴展模型"""
    def __init__(self, node_vocab, action_vocab, embedding_dim, hidden_dim):
        super(TreeExpansionModel, self).__init__()
        self.node_vocab = node_vocab
        self.action_vocab = action_vocab
        self.hidden_dim = hidden_dim
        
        self.tree_encoder = TreeLSTM(len(node_vocab), embedding_dim, hidden_dim)
        self.controller_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.action_predictor = nn.Linear(hidden_dim, len(action_vocab))

    def forward(self, current_tree_node, controller_hidden_state):
        tree_vector_h, _ = self.tree_encoder(current_tree_node, self.node_vocab)
        h, c = self.controller_lstm(tree_vector_h, controller_hidden_state)
        action_logits = self.action_predictor(h)
        return action_logits, (h, c)

class TopicChainingModel(nn.Module):
    """層級二：主題接龍模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TopicChainingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_sentence, target_length=10):
        # 編碼輸入句子
        embedded = self.embedding(input_sentence)
        _, (hidden, cell) = self.encoder(embedded)
        
        # 解碼生成種子子句
        batch_size = input_sentence.size(0)
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        
        for _ in range(target_length):
            embedded_input = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))
            prediction = self.output_projection(output)
            outputs.append(prediction)
            decoder_input = prediction.argmax(dim=-1)
        
        return torch.cat(outputs, dim=1)

class DocumentTerminationModel(nn.Module):
    """層級三：篇章終止模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DocumentTerminationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # CONTINUE=0, STOP=1
        
    def forward(self, sentence):
        embedded = self.embedding(sentence)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.classifier(hidden.squeeze(0))
        return logits

# ============================================================================
# 4. 訓練函式
# ============================================================================
def train_level1_model(model, training_sequences, node_vocab, action_vocab, 
                      num_epochs=10, lr=0.001, save_path="level1_model.pth"):
    print(f"開始訓練層級一模型（{len(training_sequences)} 個序列）")
    
    criterion = nn.CrossEntropyLoss(ignore_index=action_vocab.word2id["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i, sequence in enumerate(training_sequences):
            optimizer.zero_grad()
            
            # 初始化
            h = torch.zeros(1, model.hidden_dim)
            c = torch.zeros(1, model.hidden_dim)
            current_tree = TreeNode("S")  # 簡化的根節點
            sequence_loss = 0
            
            # 處理每個動作
            for action_str in sequence:
                action_logits, (h, c) = model(current_tree, (h, c))
                target_id = torch.tensor([action_vocab.word2id.get(action_str, 0)])
                loss = criterion(action_logits, target_id)
                sequence_loss += loss
                
            sequence_loss.backward()
            optimizer.step()
            total_loss += sequence_loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"  序列 {i+1}/{len(training_sequences)}, 損失: {sequence_loss.item():.4f}")
        
        avg_loss = total_loss / len(training_sequences)
        print(f"Epoch {epoch+1}/{num_epochs}, 平均損失: {avg_loss:.4f}")
    
    # 儲存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型已儲存至: {save_path}")

def generate_sentence(model, seed_words, node_vocab, action_vocab, max_actions=50):
    """使用訓練好的模型生成句子"""
    model.eval()
    
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_dim)
        c = torch.zeros(1, model.hidden_dim)
        current_tree = TreeNode("S")
        generated_actions = []
        
        for _ in range(max_actions):
            action_logits, (h, c) = model(current_tree, (h, c))
            predicted_action_id = torch.argmax(action_logits, dim=1).item()
            action_str = action_vocab.id2word.get(predicted_action_id, "<UNK>")
            
            generated_actions.append(action_str)
            
            # 如果遇到最終STOP，結束生成
            if action_str.startswith("STOP") and "S" in action_str:
                break
    
    return generated_actions

# ============================================================================
# 5. 主執行區塊
# ============================================================================
if __name__ == '__main__':
    print("=== 階段零：資料準備 ===")
    
    # 初始化 Stanza
    print("載入 Stanza 模型...")
    try:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', 
                            verbose=False, use_gpu=False)
        print("Stanza 載入完成")
    except Exception as e:
        print(f"Stanza 載入失敗: {e}")
        print("請先執行: stanza.download('en')")
        exit()

    # 載入資料集
    print("\n載入 CNN/DailyMail 資料集...")
    dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    
    NUM_ARTICLES = 3  # 測試用小數量
    VOCAB_PATH = "vocabularies.pkl"
    MODEL_PATH = "tree_expansion_model.pth"
    
    subset_dataset = dataset.take(NUM_ARTICLES)
    
    all_action_sequences = []
    all_seed_clauses = []
    all_nodes_words = set()
    
    print(f"處理前 {NUM_ARTICLES} 篇文章...")
    
    for idx, example in enumerate(subset_dataset):
        print(f"處理文章 {idx+1}/{NUM_ARTICLES}")
        article_text = example['article']
        
        try:
            doc = nlp(article_text)
            for sentence in doc.sentences:
                # 過濾過長或過短的句子
                if not (5 <= len(sentence.words) <= 25):
                    continue
                    
                tree = sentence.constituency
                if tree:
                    # 生成動作序列
                    actions = generate_expansion_actions(tree)
                    all_action_sequences.append(actions)
                    
                    # 提取種子子句
                    seed = extract_seed_clause(tree)
                    all_seed_clauses.append(seed)
                    
                    # 收集詞彙
                    for action in actions:
                        labels = re.findall(r"new_label=(\w+)", action)
                        contents = re.findall(r"content='([^']*)'", action)
                        all_nodes_words.update(labels)
                        all_nodes_words.update(contents)
        
        except Exception as e:
            print(f"處理文章 {idx+1} 時出錯: {e}")
            continue

    print(f"成功生成 {len(all_action_sequences)} 個動作序列")
    
    # 建立詞彙表
    print("\n=== 建立詞彙表 ===")
    node_vocab = Vocabulary("Node/Word")
    node_vocab.build_from_sequences([list(all_nodes_words)])
    
    action_vocab = Vocabulary("Action")  
    action_vocab.build_from_sequences(all_action_sequences)
    
    # 儲存詞彙表
    vocab_data = {
        'node_vocab': node_vocab,
        'action_vocab': action_vocab,
        'seed_clauses': all_seed_clauses
    }
    
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"詞彙表已儲存至: {VOCAB_PATH}")

    # 訓練模型
    if all_action_sequences:
        print("\n=== 階段一：模型訓練 ===")
        model = TreeExpansionModel(
            node_vocab=node_vocab,
            action_vocab=action_vocab, 
            embedding_dim=128,
            hidden_dim=256
        )
        
        train_level1_model(
            model, 
            all_action_sequences,
            node_vocab, 
            action_vocab,
            num_epochs=5,
            save_path=MODEL_PATH
        )
        
        print("\n=== 階段二：模型測試 ===")
        # 測試生成
        test_actions = generate_sentence(
            model, 
            ["the", "cat"], 
            node_vocab, 
            action_vocab,
            max_actions=20
        )
        
        print("生成的動作序列:")
        for i, action in enumerate(test_actions):
            print(f"{i+1:2d}: {action}")
    
    print("\n=== 系統構建完成 ===")
    print(f"模型檔案: {MODEL_PATH}")
    print(f"詞彙檔案: {VOCAB_PATH}")
    print("可使用 generate_sentence() 函式進行句子生成")
