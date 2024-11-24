// const fs = require('fs');
let selectedIDs = [];
let jsonData = {}; // Declare global variable
let matrix = [];
let output = null;
let totalMoney = null;
let newArray = null;
let players = ["Aaron Ramsdale", "Andreas Christensen", "André Onana", "Aurélien Tchouaméni", "Bruno Fernandes", "Christian Eriksen", "Cristiano Ronaldo", "Erling Haaland", "Florian Wirtz", "Jamal Musiala", "Jude Bellingham", "Karim Benzema", "Kyle Walker", "Kylian Mbappé", "Lionel Messi", "Luis Díaz", "Neymar Jr.", "Robert Lewandowski", "Vinicius Jr.", "Yassine Bounou"]
let latest_val = [45, 100, 20, 90, 540, 36, 675, 720, 63, 9, 99, 45, 360, 1800, 1260, 315, 1620, 90, 630, 90]
let mean = null;

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

    const yearInput = document.getElementById('year').value;
    const year = parseInt(yearInput, 10);
    console.log(year)
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
            mean = result["mean_vec"]
            newArray = output.map(weight => weight * totalMoney);

            console.log("New Array:", newArray);
            // =====================
            const outputContainer = document.getElementById("outputContainer");
            outputContainer.innerHTML = ''; // Clear any existing content

            newArray.forEach((value, index) => {
                // Create a table if not already created
                let table = document.getElementById('playerTable');
                if (!table) {
                    table = document.createElement('table');
                    table.id = 'playerTable';
                    table.style.borderCollapse = 'collapse';
                    table.style.width = '100%';
                    table.style.marginTop = '10px';
                    table.style.fontSize = '18px';

                    // Create the header row
                    const headerRow = document.createElement('tr');
                    ['Player', 'Value', 'Future value (in million)'].forEach(headerText => {
                        const th = document.createElement('th');
                        th.textContent = headerText;
                        th.style.border = '1px solid #000';
                        th.style.padding = '8px';
                        headerRow.appendChild(th);
                    });
                    table.appendChild(headerRow);

                    outputContainer.appendChild(table);
                }

                // Add a new row for each item
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td style="border: 1px solid #000; padding: 8px;">${players[parseInt(selectedIDs[index])]}</td>
                    <td style="border: 1px solid #000; padding: 8px;">${value.toFixed(2)}</td>
                    <td style="border: 1px solid #000; padding: 8px;">${latest_val[parseInt(selectedIDs[index])] * Math.exp(mean[index]*year)}</td>

                `;
                table.appendChild(row);
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

