package web

import (
	"html/template"
	"net/http"

	"ffdc.chat_application/pkg/database"
)

var indexHTML = `
<!DOCTYPE html>
<html>
<head>
	<title>RAG Server</title>
	<meta charset="UTF-8" />
	<style>
		body { font-family: Arial; margin: 40px; }
		.box { padding: 20px; border: 1px solid #ccc; width: 500px; }
		.chat { margin-top: 30px; padding: 10px; border: 1px solid #555; }
	</style>
</head>
<body>

<h2>RAG Server Status</h2>

<div class="box">
	<p><b>Database Loaded:</b> {{ .DBLoaded }}</p>
	<p><b>Embeddings in memory:</b> {{ .Count }}</p>
</div>

<h2>Chat Test</h2>
<div class="chat">
	<input id="msg" style="width: 300px;" placeholder="Type message..." />
	<button onclick="sendMsg()">Send</button>
	<pre id="log"></pre>
</div>

<script>
let ws = new WebSocket("ws://" + location.host + "/api/ws");

ws.onmessage = function(event) {
	document.getElementById("log").textContent += "AI: " + event.data + "\n";
};

function sendMsg() {
	let txt = document.getElementById("msg").value;
	ws.send(txt);
	document.getElementById("log").textContent += "You: " + txt + "\n";
	document.getElementById("msg").value = "";
}
</script>

</body>
</html>
`

func IndexPage(w http.ResponseWriter, embeddings []database.Embedding) {
	data := struct {
		DBLoaded bool
		Count    int
	}{
		DBLoaded: len(embeddings) > 0,
		Count:    len(embeddings),
	}

	t, err := template.New("index").Parse(indexHTML)
	if err != nil {
		http.Error(w, "Template error", 500)
		return
	}

	t.Execute(w, data)
}
