let selectedCards = [];
document.addEventListener('DOMContentLoaded', () => {

    // Add click event listener to each card
    document.querySelectorAll('.cards div').forEach((card) => {
        card.addEventListener('click', () => {
            const cardId = card.id.split('-')[1]; // Extract the number part of the ID (after the hyphen)

            if (selectedCards.includes(cardId)) {
                // If the card is already selected, deselect it
                selectedCards = selectedCards.filter((id) => id !== cardId);
                card.classList.remove('selected');
            } else if (selectedCards.length < 5) {
                // If less than 5 cards are selected, select the current card
                selectedCards.push(cardId);
                card.classList.add('selected');
            } else {
                alert("You can only select up to 5 cards.");
            }
        });
    });

    // Add event listener for the button
    document.getElementById('getIdsButton').addEventListener('click', () => {
        // console.log(selectedCards); // Output the selected card IDs
    });
});


console.log(selectedCards)
// Select the slider and value display elements
const slider = document.getElementById("value-slider");
const sliderValueDisplay = document.getElementById("slider-value");

// Variable to store the current slider value
let currentSliderValue = slider.value;

// Update the displayed value and the variable as the slider moves
slider.addEventListener("input", () => {
    currentSliderValue = slider.value; // Update the variable
    sliderValueDisplay.textContent = parseInt(currentSliderValue).toLocaleString(); // Format with commas
    // console.log("Current Slider Value:", currentSliderValue); // Debug: Print the current value
});





document.getElementById('getIdsButton').addEventListener('click', () => {
    const event = new CustomEvent('cardsSelected', { detail: [selectedCards,currentSliderValue]  });
    window.dispatchEvent(event); // Dispatch the event
});

