// const fs = require('fs');
let selectedIDs = [];
let jsonData = {}; // Declare global variable
let matrix = [];
let output = null;
let totalMoney = null;
let newArray = null;
let players=["Aaron Ramsdale","Andreas Christensen","André Onana","Aurélien Tchouaméni","Bruno Fernandes","Christian Eriksen","Cristiano Ronaldo","Erling Haaland","Florian Wirtz","Jamal Musiala","Jude Bellingham","Karim Benzema","Kyle Walker","Kylian Mbappé","Lionel Messi","Luis Díaz","Neymar Jr.","Robert Lewandowski","Vinicius Jr.","Yassine Bounou"]

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// Fetch the JSON data from the structured_returns.js file
fetch('structured_returns.json')
    .then(response => response.json()) // Parse the JSON data
    .then(data => {
        jsonData = data; // Store the JSON data in the global variable
        console.log(jsonData);  // Verify the data is loaded correctly
    })
    .catch(error => console.error('Error reading the JSON file:', error));


console.log(jsonData)


//eventlistener on click
window.addEventListener('cardsSelected', (event) => {
    selectedIDs = event.detail[0]; // Get the selected cards array from the event
    totalMoney = event.detail[1];
    // console.log(selectedIDs); // Use the selected cards
    // console.log(totalMoney); // Use the selected cards

    // Create the matrix (9x5)

    // Loop through the selected card IDs
    selectedCards.forEach(cardId => {
        // Push the corresponding array from the JSON data into the matrix
        if (jsonData.hasOwnProperty(cardId)) {
            matrix.push(jsonData[cardId]);
        }
    });


    //send matrix and get weights
    const data = { variable: matrix }; // Replace with your variable

    fetch("http://127.0.0.1:5000/receive", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(result => {
            console.log("Response from Python:", result);
            output = result["weights"];
            newArray = output.map(weight => weight * totalMoney);

            console.log("New Array:", newArray);
            // =====================
            const outputContainer = document.getElementById("outputContainer");
            outputContainer.innerHTML = ''; // Clear any existing content
            newArray.forEach((value, index) => {
                const item = document.createElement('div');
                item.textContent = `${players[parseInt(selectedIDs[index])] } : ${value.toFixed(2)}`;
                item.style.margin = '5px 0';
                item.style.fontSize = '18px';
                outputContainer.appendChild(item);
            });

        })
        .catch(error => console.error("Error:", error));

    // console.log("Selected cards:", event.detail); // Selected cards array

    // console.log("Current slider value:", sliderValue); // Use slider value

    console.log(totalMoney);
    console.log(output);


});
// console.log(matrix);
// console.log(selectedIDs);
// console.log(output);

