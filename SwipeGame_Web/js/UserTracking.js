"use strict";

var host = "http://localhost:5001";
var connection = new signalR.HubConnectionBuilder().withUrl(host + "/users").build();

var currentUsers = 0;
var usersToday = 0;

connection.on("NewUser", function (userID) {
    usersToday++;
    document.getElementById("usersToday").textContent = usersToday;
    currentUsers++;
    document.getElementById("currentUsers").textContent = currentUsers;
});

connection.on("LostUser", function (userID) {
    currentUsers--;
    document.getElementById("currentUsers").textContent = currentUsers;
});

connection.start().then(function () {
    document.getElementById("userConnectionState").innerHTML = "connected";
}).catch(function (err) {
    return console.error(err.toString());
});