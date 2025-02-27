"use strict";

var host = "http://localhost:5001";
var connection = new signalR.HubConnectionBuilder().withUrl(host + "/video").build();

connection.on("NewFrame", function (base64Image) {
    var img = document.getElementById("videoFrame");
    img.src = "data:image/jpeg;charset=utf-8;base64, " + base64Image;
});

connection.start().then(function () {
    document.getElementById("videoConnectionState").innerHTML = "connected";
}).catch(function (err) {
    return console.error(err.toString());
});