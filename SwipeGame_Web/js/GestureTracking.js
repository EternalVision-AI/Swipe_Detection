"use strict";

// Convert gesture ID to readable text
function GestureToString(gesture) {
    switch (gesture) {
        case 0: return "waving!";
        case 1: return "swiping left!";
        case 2: return "swiping right!";
        case 3: return "swiping up!";
        case 4: return "swiping down!";
        case 5: return "pushing!";
        default: return "idle";
    }
};

// WebSocket Connection Setup
const ws = new WebSocket("ws://127.0.0.1:8765");

// Handle WebSocket connection open
ws.onopen = function () {
    console.log("Connected to Python WebSocket server");
    document.getElementById("wsConnectionState").innerHTML = "Connected to WebSocket";
};

// Handle messages from the Python WebSocket server
ws.onmessage = function (event) {
    let data = JSON.parse(event.data);
    let gestureText = GestureToString(data.gesture);

    document.getElementById("gestureData").textContent = 
        "(" + data.gesture + ") User " + data.user_id + " is " + gestureText;
};

// Handle WebSocket connection close
ws.onclose = function () {
    console.log("WebSocket connection closed");
    document.getElementById("wsConnectionState").innerHTML = "Disconnected from WebSocket";
};

// Handle WebSocket errors
ws.onerror = function (error) {
    console.error("WebSocket Error:", error);
};
