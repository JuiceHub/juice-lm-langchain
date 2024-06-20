from fastapi import FastAPI
# from service.chat.chain import chains
# from service.chat.session import session
# from service.chat.model import model
# from service.data_connection.source import source

app = FastAPI()


# app.include_router(chains, prefix='/chain')
# app.include_router(session, prefix='/session')
# app.include_router(model, prefix='/model')
# app.include_router(source, prefix='/data_connection/source')

@app.get("/")
async def root():
    return {"message": "Hello World"}
