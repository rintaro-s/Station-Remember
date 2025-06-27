# app.py (UI・検索方法刷新版)

import streamlit as st
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate

# --- 設定項目 ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
FAISS_INDEX_PATH = "faiss_index"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# --- アプリの基本設定 ---
st.set_page_config(page_title="Station Remember - 駅メモWikiチャットボット", layout="wide")

# --- サイドバー ---
with st.sidebar:
    st.header("Station Remember - 駅メモWikiチャットボット")
    st.write("このチャットボットは、駅メモ！wikiの情報を元に、あなたの質問にAIが回答します。")
    st.write("---")
    st.header("使い方")
    st.info("下のチャット入力欄に質問を入力してください。AIがWikiの情報を検索・分析し、その思考プロセスと共に回答を生成します。")
    if st.button("チャット履歴をクリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- RAG Agentのセットアップ ---
@st.cache_resource
def load_agent():
    """FAISSのみで検索するAgentをロードします。"""
    try:
        # FAISS (ベクトル検索) Retrieverの準備
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # ← ここを20に変更
    except Exception as e:
        st.error(f"データベースの読み込みに失敗しました。`prepare.py`を正常に実行しましたか？\nエラー: {e}")
        return None

    llm = ChatOpenAI(base_url=LMSTUDIO_BASE_URL, api_key="not-needed", temperature=0.2, streaming=True)

    retriever_tool = create_retriever_tool(
        faiss_retriever,  # FAISSのみ
        name="faiss_ekimemo_wiki_search",
        description="駅メモ！のでんこ、スキル、イベント、編成などに関する情報をベクトル検索で探します。"
    )
    tools = [retriever_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
あなたは「駅メモ！」の専門家AIアシスタントです。
ユーザーの質問に答えるために、提供されたツールを駆使して情報を収集・分析し、統合して最終的な回答を生成してください。

### 行動原則:
1. **段階的調査:** 複雑な質問は、複数のステップに分解して、段階的にツールを使って調査してください。
2. **情報集約と結論:** ツールを複数回使用して情報を十分に収集した後は、**必ず全ての情報を統合し、最終的な回答を生成して思考を終了してください。**
3. **自己修正:** もし同じような検索を繰り返していることに気づいた場合は、検索キーワードを変えるか、すでにある情報だけで回答を作成してください。**無限に検索を続けないでください。**
4. **日本語での回答:** 回答は必ず日本語で行ってください。
5. **不明な場合:** Wikiの情報にないことは「分かりません」と正直に回答してください。
6. **簡潔な回答:** ユーザーに最終的な回答を返す時、思考プロセスやツール使用について言及する必要はありません。結論だけを述べてください。
"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
    return agent_executor

# --- メイン画面 ---
st.title("チャット")

agent_executor = load_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("「でんこ」や編成について質問してください"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        final_answer = ""
        workflow_steps = []
        think_steps = []

        # 思考プロセスと最終回答の表示場所を明確に分離
        workflow_placeholder = st.empty()
        answer_placeholder = st.empty()
        think_placeholder = st.empty()

        try:
            with st.spinner("AIが思考中です..."):
                response_generator = agent_executor.stream(
                    {"input": user_question, "chat_history": st.session_state.messages}
                )

                for chunk in response_generator:
                    # <think>パートの抽出
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            # <think>タグが含まれている場合は分離
                            tool_input = str(action.tool_input)
                            if "<think>" in tool_input and "</think>" in tool_input:
                                before, think_part = tool_input.split("<think>", 1)
                                think_content, after = think_part.split("</think>", 1)
                                workflow_steps.append(f"**検索実行:** `{before.strip() + after.strip()}`")
                                think_steps.append(think_content.strip())
                            else:
                                workflow_steps.append(f"**検索実行:** `{tool_input}`")
                    elif "steps" in chunk:
                        workflow_steps.append("**結果分析:** (関連情報を取得し、次の行動を検討中...)")
                    # <think>パートがoutputに含まれる場合も分離
                    if "output" in chunk:
                        output = chunk.get("output", "")
                        if "<think>" in output and "</think>" in output:
                            before, think_part = output.split("<think>", 1)
                            think_content, after = think_part.split("</think>", 1)
                            final_answer = before.strip() + after.strip()
                            think_steps.append(think_content.strip())
                        else:
                            final_answer = output

                    # 思考プロセスをリアルタイムで更新
                    with workflow_placeholder.expander("AIの思考プロセスを見る", expanded=True):
                        st.markdown("\n\n---\n\n".join(f"- {s}" for s in workflow_steps))
                    # <think>パートを別枠で表示・非表示切替
                    if think_steps:
                        with think_placeholder.expander("AIの<think>パート（思考の詳細）", expanded=False):
                            st.markdown("\n\n---\n\n".join(think_steps))

        except Exception as e:
            final_answer = f"申し訳ありません、処理中にエラーが発生しました。\n`{e}`"

        if not final_answer:
            final_answer = "申し訳ありません、思考がループしたか、時間内に結論に達しませんでした。質問を変えるか、より具体的にしてみてください。"

        answer_placeholder.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})