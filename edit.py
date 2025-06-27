# edit.py - 荒らしページ除外プログラム

import re
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- 設定項目 ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
FAISS_INDEX_PATH = "faiss_index"
CLEANED_FAISS_INDEX_PATH = "faiss_index_cleaned"

def is_spam_page(content, title, url):
    """
    荒らしページかどうかを判定する
    """
    # 1. 内容が極端に短い（50文字以下）
    if len(content.strip()) < 50:
        print(f"短すぎる内容: {title}")
        return True
    
    # 2. 意味のない文字列パターンを検出
    # ランダムな文字列の特徴を検出
    nonsense_patterns = [
        r'[あ-ん]{1}[A-Za-z0-9]{1,3}[あ-ん]{1}[A-Za-z0-9]{1,3}',  # あaいbうc のようなパターン
        r'[ア-ン]{1}[A-Za-z0-9]{1,3}[ア-ン]{1}[A-Za-z0-9]{1,3}',  # アaイbウc のようなパターン
        r'[あ-んア-ン]{3,}[A-Za-z0-9]{1,3}[あ-んア-ン]{3,}',      # ランダムな文字列
    ]
    
    nonsense_count = 0
    for pattern in nonsense_patterns:
        matches = re.findall(pattern, content)
        nonsense_count += len(matches)
    
    # 意味のない文字列が多い場合（全文字数の30%以上）
    if nonsense_count > len(content) * 0.3:
        print(f"意味のない文字列が多い: {title} (nonsense_count: {nonsense_count})")
        return True
    
    # 3. URLに意味のない文字列が含まれている
    if len(re.findall(r'%[0-9A-Fa-f]{2}', url)) > 20:  # URLエンコードが異常に多い
        print(f"URLに異常な文字列: {title}")
        return True
    
    # 4. 同じ文字の繰り返しが多い
    repeated_chars = re.findall(r'(.)\1{5,}', content)  # 同じ文字が6回以上連続
    if len(repeated_chars) > 0:
        print(f"同じ文字の繰り返しが多い: {title}")
        return True
    
    # 5. 日本語の文章として不自然（ひらがな・カタカナ・漢字の比率）
    hiragana_count = len(re.findall(r'[あ-ん]', content))
    katakana_count = len(re.findall(r'[ア-ン]', content))
    kanji_count = len(re.findall(r'[一-龯]', content))
    ascii_count = len(re.findall(r'[A-Za-z0-9]', content))
    
    total_chars = len(content)
    if total_chars > 0:
        japanese_ratio = (hiragana_count + katakana_count + kanji_count) / total_chars
        if japanese_ratio < 0.3:  # 日本語の割合が30%未満
            print(f"日本語の割合が低い: {title} (ratio: {japanese_ratio:.2f})")
            return True
    
    # 6. 特定のスパムキーワードを含む
    spam_keywords = [
        "最終更新:2025-05-03",  # 同じ日付で大量作成されたページ
        "ページトップ...",      # 実質的な内容がない
    ]
    
    for keyword in spam_keywords:
        if keyword in content:
            print(f"スパムキーワードを含む: {title}")
            return True
    
    return False

def filter_spam_documents():
    """
    既存のFAISSインデックスから荒らしページを除外した新しいインデックスを作成
    """
    print("FAISSインデックスの読み込み中...")
    
    # 既存のインデックスを読み込み
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"エラー: FAISSインデックスの読み込みに失敗しました。{e}")
        return
    
    # すべてのドキュメントを取得
    print("ドキュメントの取得中...")
    
    # FAISSから直接ドキュメントを取得する方法
    # vectorstore.docstore.mget で全ドキュメントを取得
    all_ids = list(vectorstore.docstore._dict.keys())
    all_documents = []
    
    for doc_id in all_ids:
        doc = vectorstore.docstore._dict[doc_id]
        all_documents.append(doc)
    
    print(f"合計 {len(all_documents)} 件のドキュメントを取得しました。")
    
    # スパムページを除外
    clean_documents = []
    spam_count = 0
    
    for doc in all_documents:
        content = doc.page_content
        title = doc.metadata.get('title', 'タイトル不明')
        url = doc.metadata.get('source', '')
        
        if not is_spam_page(content, title, url):
            clean_documents.append(doc)
        else:
            spam_count += 1
            print(f"除外: {title}")
    
    print(f"\n除外されたページ数: {spam_count}")
    print(f"残ったページ数: {len(clean_documents)}")
    
    if len(clean_documents) == 0:
        print("警告: すべてのドキュメントが除外されました。フィルタリング条件を確認してください。")
        return
    
    # 新しいFAISSインデックスを作成
    print("新しいFAISSインデックスを作成中...")
    cleaned_vectorstore = FAISS.from_documents(clean_documents, embeddings)
    
    # クリーンなインデックスを保存
    cleaned_vectorstore.save_local(CLEANED_FAISS_INDEX_PATH)
    print(f"クリーンなFAISSインデックスを '{CLEANED_FAISS_INDEX_PATH}' に保存しました。")
    
    # 統計情報を保存
    stats = {
        "original_count": len(all_documents),
        "cleaned_count": len(clean_documents),
        "spam_count": spam_count,
        "spam_rate": spam_count / len(all_documents) * 100
    }
    
    with open("cleaning_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n統計情報:")
    print(f"- 元のページ数: {stats['original_count']}")
    print(f"- クリーンなページ数: {stats['cleaned_count']}")
    print(f"- 除外されたページ数: {stats['spam_count']}")
    print(f"- スパム率: {stats['spam_rate']:.2f}%")

def filter_spam_documents_with_user_input():
    """
    スパム候補をリストアップし、ユーザーが除外するページ番号を選択してクリーンなインデックスを作成
    """
    print("FAISSインデックスの読み込み中...")

    # 既存のインデックスを読み込み
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"エラー: FAISSインデックスの読み込みに失敗しました。{e}")
        return

    # すべてのドキュメントを取得
    print("ドキュメントの取得中...")
    all_ids = list(vectorstore.docstore._dict.keys())
    all_documents = []

    for doc_id in all_ids:
        doc = vectorstore.docstore._dict[doc_id]
        all_documents.append(doc)

    print(f"合計 {len(all_documents)} 件のドキュメントを取得しました。")

    # スパム候補をリストアップ
    spam_candidates = []
    for doc in all_documents:
        content = doc.page_content
        title = doc.metadata.get('title', 'タイトル不明')
        url = doc.metadata.get('source', '')

        if is_spam_page(content, title, url):
            spam_candidates.append(doc)

    print(f"スパム候補: {len(spam_candidates)} 件")

    for i, doc in enumerate(spam_candidates):
        print(f"\n--- スパム候補 {i+1} ---")
        print(f"タイトル: {doc.metadata.get('title', 'N/A')}")
        print(f"URL: {doc.metadata.get('source', 'N/A')}")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"文字数: {len(doc.page_content)}")

    # ユーザーに除外するページ番号を入力させる
    exclude_indices = input("除外するページ番号をカンマ区切りで入力してください: ")
    try:
        exclude_indices = [int(idx.strip()) - 1 for idx in exclude_indices.split(",")]
    except ValueError:
        print("無効な入力です。処理を終了します。")
        return

    # 除外対象を除いたクリーンなドキュメントを作成
    clean_documents = []
    for i, doc in enumerate(all_documents):
        if i not in exclude_indices:
            clean_documents.append(doc)

    print(f"\n除外されたページ数: {len(exclude_indices)}")
    print(f"残ったページ数: {len(clean_documents)}")

    if len(clean_documents) == 0:
        print("警告: すべてのドキュメントが除外されました。フィルタリング条件を確認してください。")
        return

    # 新しいFAISSインデックスを作成
    print("新しいFAISSインデックスを作成中...")
    cleaned_vectorstore = FAISS.from_documents(clean_documents, embeddings)

    # クリーンなインデックスを保存
    cleaned_vectorstore.save_local(CLEANED_FAISS_INDEX_PATH)
    print(f"クリーンなFAISSインデックスを '{CLEANED_FAISS_INDEX_PATH}' に保存しました。")

    # 統計情報を保存
    stats = {
        "original_count": len(all_documents),
        "cleaned_count": len(clean_documents),
        "spam_count": len(exclude_indices),
        "spam_rate": len(exclude_indices) / len(all_documents) * 100
    }

    with open("cleaning_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n統計情報:")
    print(f"- 元のページ数: {stats['original_count']}")
    print(f"- クリーンなページ数: {stats['cleaned_count']}")
    print(f"- 除外されたページ数: {stats['spam_count']}")
    print(f"- スパム率: {stats['spam_rate']:.2f}%")

def manual_check_documents():
    """
    手動でドキュメントをチェックするためのヘルパー関数
    """
    print("手動チェック用: 短いドキュメントの一覧")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"エラー: FAISSインデックスの読み込みに失敗しました。{e}")
        return
    
    all_ids = list(vectorstore.docstore._dict.keys())
    
    short_docs = []
    for doc_id in all_ids:
        doc = vectorstore.docstore._dict[doc_id]
        if len(doc.page_content) < 100:  # 100文字以下
            short_docs.append(doc)
    
    print(f"100文字以下のドキュメント: {len(short_docs)}件")
    
    for i, doc in enumerate(short_docs[:10]):  # 最初の10件を表示
        print(f"\n--- ドキュメント {i+1} ---")
        print(f"タイトル: {doc.metadata.get('title', 'N/A')}")
        print(f"URL: {doc.metadata.get('source', 'N/A')}")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"文字数: {len(doc.page_content)}")

if __name__ == "__main__":
    print("荒らしページ除外プログラム")
    print("1. スパムページを除外してクリーンなインデックスを作成")
    print("2. 手動チェック（短いドキュメントの確認）")
    print("3. スパム候補を提示してユーザー選択で除外")
    
    choice = input("選択してください (1/2/3): ")
    
    if choice == "1":
        filter_spam_documents()
    elif choice == "2":
        manual_check_documents()
    elif choice == "3":
        filter_spam_documents_with_user_input()
    else:
        print("無効な選択です。")
