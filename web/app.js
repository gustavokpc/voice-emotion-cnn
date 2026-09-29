const record = document.querySelector('#record'), stop = document.querySelector('#stop');
const status = document.querySelector('#status'), file = document.querySelector('#file'), model = document.querySelector('#model'); let recorder, chunks = [];
record.onclick = async () => { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); recorder = new MediaRecorder(stream); chunks = []; recorder.ondataavailable = e => chunks.push(e.data); recorder.onstop = () => send(new File(chunks, 'recording.webm', { type: recorder.mimeType })); recorder.start(); record.disabled = true; stop.disabled = false; status.textContent = 'Recording...'; };
stop.onclick = () => { recorder.stop(); recorder.stream.getTracks().forEach(t => t.stop()); record.disabled = false; stop.disabled = true; };
file.onchange = () => file.files[0] && send(file.files[0]);
async function send(audio) {
  status.textContent = 'Analyzing... The first request may take up to a minute while the model starts.';
  document.querySelector('#result').hidden = true;
  const body = new FormData();
  body.append('audio', audio);
  body.append('architecture', model.value);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 90_000);
  try {
    const response = await fetch('/predict', { method: 'POST', body, signal: controller.signal });
    const data = response.headers.get('content-type')?.includes('application/json')
      ? await response.json()
      : {};
    if (!response.ok) {
      status.textContent = data.detail || `Analysis failed (server returned ${response.status}).`;
      return;
    }
    status.textContent = 'Analysis complete.';
    document.querySelector('#result').hidden = false;
    document.querySelector('#emotion').textContent = `Emotion: ${data.emotion}`;
    document.querySelector('#confidence').textContent = `Model: ${data.architecture} · Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    drawSpectrogram(data.spectrogram);
    document.querySelector('#scores').innerHTML = Object.entries(data.scores)
      .map(([label, score]) => `<li>${label}: ${(score * 100).toFixed(1)}%</li>`)
      .join('');
  } catch (error) {
    status.textContent = error.name === 'AbortError'
      ? 'Analysis timed out after 90 seconds. The server may be starting or out of resources.'
      : 'Could not reach the analysis server. Please try again in a moment.';
  } finally {
    window.clearTimeout(timeout);
  }
}
function drawSpectrogram(mel) { const c = document.querySelector('#spectrogram'), ctx = c.getContext('2d'), h = mel.length, w = mel[0].length; const min = Math.min(...mel.flat()), max = Math.max(...mel.flat()); const image = ctx.createImageData(c.width, c.height); for (let y=0;y<c.height;y++) for (let x=0;x<c.width;x++) { const v=(mel[h-1-Math.floor(y*h/c.height)][Math.floor(x*w/c.width)]-min)/(max-min||1), p=(y*c.width+x)*4; image.data[p]=Math.floor(25+220*v); image.data[p+1]=Math.floor(40+150*v); image.data[p+2]=Math.floor(100+100*(1-v)); image.data[p+3]=255; } ctx.putImageData(image,0,0); }
