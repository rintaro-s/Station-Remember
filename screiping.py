# prepare.py (改良版)

import requests
from bs4 import BeautifulSoup
import time
import os
import pickle
import json
from lxml import etree
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- 設定項目 ---
# 1. サイトマップのURL
WIKI_SITEMAP_URL = "https://ek1mem0.wiki.fc2.com/sitemap.xml"

# 2. 埋め込みに使用するローカルモデル名
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 

# 3. FAISSインデックスの保存先ディレクトリ名
FAISS_INDEX_PATH = "faiss_index"

# 4. サイトへの負荷対策設定
REQUEST_DELAY = 1.0  # 各リクエストの間に1.0秒の待機を入れる
HTTP_HEADERS = {
    'User-Agent': 'WikiChatbotScraper/1.0 (for personal AI project)'
}

# 5. 状態保存用のファイルパス
STATE_FILES = {
    "documents": "documents.pkl",
    "metadata": "metadata.json"
}

# --- 関数定義 ---

def load_sitemap(url):
    """サイトマップを解析し、URLと最終更新日の辞書を返す"""
    print(f"サイトマップを取得中: {url}")
    try:
        response = requests.get(url, headers=HTTP_HEADERS)
        response.raise_for_status()
        sitemap_xml = etree.fromstring(response.content)
        
        # XML名前空間を取得
        ns = {'s': sitemap_xml.nsmap.get(None, '')}
        
        url_data = {}
        for url_element in sitemap_xml.xpath('//s:url', namespaces=ns):
            loc = url_element.xpath('s:loc/text()', namespaces=ns)[0]
            lastmod = url_element.xpath('s:lastmod/text()', namespaces=ns)
            url_data[loc] = lastmod[0] if lastmod else 'N/A'
        
        print(f"サイトマップから {len(url_data)} 件のURLを検出しました。")
        return url_data
    except requests.RequestException as e:
        print(f"エラー: サイトマップの取得に失敗しました。 {e}")
        return {}
    except etree.XMLSyntaxError as e:
        print(f"エラー: サイトマップの解析に失敗しました。 {e}")
        return {}

def load_local_state():
    """ローカルに保存されたドキュメントとメタデータを読み込む"""
    if os.path.exists(STATE_FILES["documents"]) and os.path.exists(STATE_FILES["metadata"]):
        print("ローカルに保存された状態ファイルを読み込んでいます...")
        with open(STATE_FILES["documents"], "rb") as f:
            local_docs = pickle.load(f)
        with open(STATE_FILES["metadata"], "r") as f:
            local_meta = json.load(f)
        return local_docs, local_meta
    return [], {}

def save_local_state(docs, meta):
    """現在のドキュメントとメタデータをローカルに保存する"""
    print("現在の状態をローカルファイルに保存しています...")
    with open(STATE_FILES["documents"], "wb") as f:
        pickle.dump(docs, f)
    with open(STATE_FILES["metadata"], "w") as f:
        json.dump(meta, f, indent=2)

def scrape_single_page(url):
    """単一のページをスクレイピングしてDocumentオブジェクトを返す"""
    try:
        response = requests.get(url, headers=HTTP_HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.select_one('div#main')
        
        if main_content:
            title = soup.select_one('h2.title').get_text(strip=True) if soup.select_one('h2.title') else "無題"
            content_text = main_content.get_text(separator='\n', strip=True)
            return Document(
                page_content=content_text,
                metadata={'source': url, 'title': title}
            )
    except requests.RequestException as e:
        print(f"  - エラー: {url} の取得に失敗。スキップします。 ({e})")
    return None

# --- メイン処理 ---
if __name__ == "__main__":
    # 1. サイトマップとローカルの状態を読み込む
    sitemap_data = load_sitemap(WIKI_SITEMAP_URL)
    local_docs, local_meta = load_local_state()

    # 2. 更新が必要なURLを特定する
    urls_to_update = []
    for url, lastmod in sitemap_data.items():
        if url not in local_meta or local_meta.get(url) != lastmod:
            urls_to_update.append(url)
            
    print(f"\n差分チェックの結果: {len(urls_to_update)} 件の新規・更新ページが見つかりました。")

    if not urls_to_update and os.path.exists(FAISS_INDEX_PATH):
        print("すべてのデータは最新です。処理を終了します。")
    else:
        # 3. 変更があったページのみスクレイピング
        if urls_to_update:
            print("ページの取得と更新を開始します...")
            updated_docs_map = {doc.metadata['source']: doc for doc in local_docs}
            
            for i, url in enumerate(urls_to_update):
                print(f"  [{i+1}/{len(urls_to_update)}] {url} を取得中...")
                new_doc = scrape_single_page(url)
                if new_doc:
                    updated_docs_map[url] = new_doc
                time.sleep(REQUEST_DELAY) # 負荷対策
            
            # サイトマップに存在するページのみに絞り込む (削除されたページを除外)
            final_docs = [doc for url, doc in updated_docs_map.items() if url in sitemap_data]
            print("ページの更新が完了しました。")
        else:
            final_docs = local_docs

        # 4. テキストをチャンクに分割
        print("\nテキストのチャンク分割を開始します...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunked_docs = text_splitter.split_documents(final_docs)
        print(f"チャンク分割が完了しました。合計 {len(chunked_docs)} チャンク。")

        # 5. FAISSインデックスを再構築・保存
        if chunked_docs:
            print("\nFAISSインデックスの構築を開始します...")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore = FAISS.from_documents(chunked_docs, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"FAISSインデックスを '{FAISS_INDEX_PATH}' に保存しました。")
            
            # 6. 最新の状態でローカルファイルを更新
            save_local_state(final_docs, sitemap_data)
        else:
            print("処理するドキュメントがないため、インデックスは作成されませんでした。")

        print("\n事前準備がすべて完了しました。 `app.py` を実行してください。")