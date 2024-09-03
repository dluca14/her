from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time

from history_rag import get_chain

app = FastAPI()

# Your chatbot's response generation function
def generate_response(prompt: str):
    # Simulate a chatbot response that generates text in chunks
    responses = [
        "Hello! ",
        "How can I assist you today? ",
        "I'm here to help with any questions you have.",
    ]
    for response in responses:
        time.sleep(1)  # Simulate delay in generating response
        yield response

@app.post("/chat")
async def chat(prompt: str):
    # Use the generator function to stream the response
    chain = get_chain()
    response = chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )['response']

    return StreamingResponse(response, media_type="text/plain")

import gradio as gr
# Create the Gradio Interface
iface = gr.Interface(
    fn=chat,  # The function to call for the chatbot response
    inputs="text",     # The input type is a single text box
    outputs="text",    # The output is also text
    title="Chat with AI",
    description="Enter a prompt to chat with the AI bot."
)

# Launch the interface
iface.launch()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
