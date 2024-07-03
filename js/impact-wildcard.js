import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let refresh_btn = document.getElementById('comfy-refresh-button');
let refresh_btn2 = document.querySelector('button[title="Refresh widgets in nodes to find new models or files"]');

let orig = refresh_btn.onclick;

refresh_btn.onclick = function() {
	orig();
	api.fetchApi('/impact/wildcards/refresh');
};

refresh_btn2.addEventListener('click', function() {
	api.fetchApi('/impact/wildcards/refresh');
});