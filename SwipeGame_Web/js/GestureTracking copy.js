"use strict";

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

var host = "http://localhost:5001";
var connection = new signalR.HubConnectionBuilder().withUrl(host + "/gestures").build();

connection.on("NewGesture", function (gesture, userId) {
    document.getElementById("gestureData").textContent = "(" + gesture + ") User " + userId + " is " + GestureToString(gesture);
    
});

connection.start().then(function () {
    document.getElementById("gestureConnectionState").innerHTML = "connected";
}).catch(function (err) {
    return console.error(err.toString());
});