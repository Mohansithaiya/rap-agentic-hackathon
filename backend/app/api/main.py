"""FastAPI entry point for inventory and document-answering interfaces."""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent.agent import InventoryAgent, InventoryResponse
from app.documents.agent import DocumentAgent
from app.documents.harness import DocumentHarness
from app.documents.store import document_store


class AskRequest(BaseModel):
    """User message sent to the inventory agent."""

    message: str


app = FastAPI()


@app.get("/documents", response_class=HTMLResponse)
def document_chat() -> str:
    """Serve the lightweight PDF chat interface."""
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAP · Document Intelligence</title><style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#192235;background:#f3f6fa}*{box-sizing:border-box}body{margin:0}.shell{max-width:920px;margin:38px auto;padding:0 20px}.brand{font-size:13px;font-weight:700;letter-spacing:.12em;color:#48617e;text-transform:uppercase}.title{font-size:30px;margin:10px 0 5px}.sub{color:#68768b;margin:0 0 24px}.card{background:white;border:1px solid #dfe6ef;border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 4px 18px #18304a0a}.upload{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.upload input{flex:1;min-width:180px}.button{border:0;border-radius:9px;padding:11px 16px;background:#205ac4;color:white;font-weight:650;cursor:pointer}.button:disabled{opacity:.55;cursor:wait}.status{font-size:13px;color:#627188;margin-top:10px}.chat{min-height:145px;display:flex;flex-direction:column;gap:14px;margin-bottom:16px}.bubble{padding:14px 16px;border-radius:12px;max-width:88%;line-height:1.55;white-space:pre-wrap}.user{align-self:flex-end;background:#eaf1ff}.answer{align-self:flex-start;background:#f4f6f9}.answer.insufficient{border-left:3px solid #d59a28}.question{display:flex;gap:10px}.question textarea{flex:1;resize:vertical;min-height:48px;border:1px solid #d7e0eb;border-radius:9px;padding:12px;font:inherit}.meta{font-size:13px;color:#64738a;margin-top:10px}.trace{margin-top:14px;border-top:1px solid #e6ebf1;padding-top:10px}.trace summary{cursor:pointer;font-weight:650;font-size:14px}.trace ol{padding-left:22px;color:#526177;font-size:13px}.fail{color:#a4423e}.pill{display:inline-block;border-radius:20px;padding:3px 9px;background:#eaf1ff;color:#2858a1;font-size:12px}@media(max-width:600px){.shell{margin:22px auto}.title{font-size:25px}.card{padding:15px}}
</style></head><body><main class="shell"><div class="brand">RAP · AI/ML Hackathon</div><h1 class="title">Document Intelligence</h1><p class="sub">Ask grounded questions about a PDF, with cited pages and a transparent tool trace.</p>
<section class="card"><form id="upload" class="upload"><input id="pdf" type="file" accept="application/pdf" required><button class="button">Upload PDF</button></form><div id="status" class="status">Choose a PDF to begin.</div></section>
<section class="card"><div id="chat" class="chat"><div class="bubble answer">Your document answers will appear here.</div></div><form id="ask" class="question"><textarea id="question" placeholder="Ask a question about your PDF…" required></textarea><button id="askButton" class="button" disabled>Ask</button></form><div id="details"></div></section></main>
<script>let docId=null;const status=document.querySelector('#status'),chat=document.querySelector('#chat'),details=document.querySelector('#details'),askButton=document.querySelector('#askButton');
document.querySelector('#upload').addEventListener('submit',async e=>{e.preventDefault();const f=document.querySelector('#pdf').files[0];if(!f)return;status.textContent='Loading PDF…';askButton.disabled=true;try{const r=await fetch('/documents/upload',{method:'POST',headers:{'Content-Type':'application/pdf','X-Filename':encodeURIComponent(f.name)},body:f});const d=await r.json();if(!r.ok)throw Error(d.detail||'Upload failed');docId=d.doc_id;status.textContent=`Ready: ${d.title} · ${d.page_count} pages`;askButton.disabled=false;chat.innerHTML='';details.innerHTML=''}catch(err){status.textContent=err.message}});
document.querySelector('#ask').addEventListener('submit',async e=>{e.preventDefault();if(!docId)return;const q=document.querySelector('#question').value.trim();if(!q)return;const user=document.createElement('div');user.className='bubble user';user.textContent=q;chat.append(user);document.querySelector('#question').value='';askButton.disabled=true;status.textContent='Researching your document…';try{const r=await fetch('/documents/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,doc_id:docId})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Could not answer question');const a=document.createElement('div');a.className='bubble answer'+(d.answer==='Insufficient information.'?' insufficient':'');a.textContent=d.answer;chat.append(a);const pages=[...new Set((d.answer.match(/\\[(?:p\\.|page)\\s*\\d+\\]/gi)||[]).map(x=>x.match(/\\d+/)[0]))];const trace=(d.trace||[]).map(t=>`<li class="${t.success?'':'fail'}"><strong>${escapeHtml(t.tool_name)}</strong> · ${t.success?'completed':t.blocked?'blocked':'failed'}${t.tool_call_number?` · call ${t.tool_call_number}/6`:''}${t.error?` · ${escapeHtml(safeFailure(t.error))}`:''}</li>`).join('');const diagnostic=(d.trace||[]).map(t=>t.result_metadata&&t.result_metadata.diagnostics?`<details><summary>Final-answer diagnostics</summary><pre>${escapeHtml(JSON.stringify(t.result_metadata.diagnostics,null,2))}</pre></details>`:'').join('');details.innerHTML=`<div class="meta"><span class="pill">${d.calls_made} / 6 tool calls</span>${pages.length?` &nbsp; Cited pages: ${pages.map(p=>'p. '+p).join(', ')}`:''}${d.answer==='Insufficient information.'?' &nbsp; No supported answer found.':''}</div><details class="trace"><summary>Tool trace (${d.trace.length})</summary><ol>${trace||'<li>No tool calls</li>'}</ol>${diagnostic}</details>`;chat.scrollTop=chat.scrollHeight;status.textContent='Answer ready.'}catch(err){status.textContent=err.message}finally{askButton.disabled=false}});
function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function safeFailure(e){return /not allowed|duplicate|maximum/i.test(e)||['localhost','127.0.0.1','::1'].includes(location.hostname)?e:'Tool could not complete.'}</script></body></html>"""


@app.post("/documents/upload")
async def upload_document(request: Request) -> dict[str, object]:
    """Load a browser-uploaded PDF using the existing document store loader."""
    if "application/pdf" not in request.headers.get("content-type", "").lower():
        raise HTTPException(status_code=415, detail="Upload a PDF file.")
    content = await request.body()
    if not content or not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    filename = unquote(request.headers.get("x-filename", "document.pdf"))
    stem = Path(filename).name[:120] or "document.pdf"
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "upload.pdf"
            temp_path.write_bytes(content)
            metadata = document_store.register_pdf(temp_path)
        metadata.title = Path(stem).stem or metadata.title
    except Exception:
        raise HTTPException(status_code=400, detail="The PDF could not be read.") from None
    return metadata.model_dump()


class DocumentAskRequest(BaseModel):
    question: str
    doc_id: str


@app.post("/documents/ask")
def ask_document(request: DocumentAskRequest) -> dict[str, object]:
    """Answer against one selected PDF and return its trace and budget usage."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Enter a question.")
    try:
        document_store.get(request.doc_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again.") from None
    try:
        result = DocumentAgent(DocumentHarness()).run(request.question, document_id=request.doc_id)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Document answering is not configured.") from None
    return result.model_dump()


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "healthy"}


@app.post("/ask", response_model=InventoryResponse)
def ask(request: AskRequest) -> InventoryResponse:
    """Pass a user message to the inventory agent."""
    agent = InventoryAgent()
    return agent.run(request.message)
