"use strict";

function ShowHand(pointer, data) {
    if (pointer == null)
        return;

    pointer.style.display = "block";
    pointer.style.top = (data.y * 100) + "%";
    pointer.style.left = (data.x * 100) + "%";
};

function HideHand(pointer) {
    if (pointer == null)
        return;

    pointer.style.display = "none";
}

function SpawnHands(containerId) {
    var handContainer = document.getElementById(containerId);

    if (handContainer == null)
        return;

    leftHandPointer = document.createElement("div");
    leftHandPointer.className = "hand";
    leftHandPointer.id = "leftHand";
    handContainer.appendChild(leftHandPointer);

    rightHandPointer = document.createElement("div");
    rightHandPointer.className = "hand";
    rightHandPointer.id = "rightHand";
    handContainer.appendChild(rightHandPointer);
}

var host = "http://localhost:5001";
var connection = new signalR.HubConnectionBuilder().withUrl(host + "/handtracking").build();

var leftHandPointer = null;
var rightHandPointer = null;

connection.on("Update", function (data) {
    if (data.leftHand != null) {
        ShowHand(leftHandPointer, data.leftHand);
        document.getElementById("leftHandData").innerHTML = JSON.stringify(data.leftHand);
    } else {
        HideHand(leftHandPointer);
        document.getElementById("leftHandData").innerHTML = "none";
    }

    if (data.rightHand != null) {
        ShowHand(rightHandPointer, data.rightHand);
        document.getElementById("rightHandData").innerHTML = JSON.stringify(data.rightHand);
    } else {
        HideHand(rightHandPointer);
        document.getElementById("rightHandData").innerHTML = "none";
    }
});

connection.on("OnTouch", function (userId, handIndex, x, y) {
    if (handIndex === 0) {
        leftHandPointer.style.backgroundColor = "#FFFFFF";
    } else {
        rightHandPointer.style.backgroundColor = "#FFFFFF";
    }
});

connection.on("OnRelease", function (userId, handIndex, x, y) {
    if (handIndex === 0) {
        leftHandPointer.style.backgroundColor = "transparent";
    } else {
        rightHandPointer.style.backgroundColor = "transparent";
    }
});

connection.start().then(function () {
    document.getElementById("handTrackingConnectionState").innerHTML = "connected";
    SpawnHands("handsContainer");
}).catch(function (err) {
    return console.error(err.toString());
});