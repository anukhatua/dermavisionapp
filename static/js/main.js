function showScanner(){
  const s = document.getElementById('scanner-overlay');
  if(s) s.style.display = 'flex';
  // auto-hide after 3.5s if server responds earlier it will refresh page
  setTimeout(()=>{ if(s) s.style.display='none'; }, 6000);
}

function downloadJSON(obj){
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(obj, null, 2));
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute('href', dataStr);
  const ts = new Date().toISOString().replace(/[:.]/g,'-');
  dlAnchor.setAttribute('download', `dermavision_result_${ts}.json`);
  document.body.appendChild(dlAnchor);
  dlAnchor.click();
  dlAnchor.remove();
}
