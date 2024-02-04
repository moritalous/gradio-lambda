import gradio as gr
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import BedrockChat

llm = BedrockChat(
    model_id="anthropic.claude-instant-v1",
    streaming=True,
)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    response = llm.stream(history_langchain_format)

    chunks = None
    for chunk in response:
        chunks = chunks + chunk if chunks is not None else chunk
        yield chunks.content


if __name__ == "__main__":
    gr.ChatInterface(predict).launch(
        server_port=8000, debug=True, show_error=True, show_api=False
    )
