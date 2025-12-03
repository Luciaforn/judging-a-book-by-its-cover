document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("classify-form");
    if (!form) return;

    const fileInput = document.getElementById("file-input");
    const urlInput = document.querySelector(".url-input");
    const mainButton = document.querySelector(".primary-button");

    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("preview-img");
    const loadingOverlay = document.getElementById("loading-overlay");

    const resultsArea = document.getElementById("results-area");
    const top1Label = document.getElementById("result-top1-label");
    const top2Label = document.getElementById("result-top2-label");
    const top3Label = document.getElementById("result-top3-label");

    let isSubmitting = false;

    // Keep mutual exclusivity: picking file clears URL, typing URL clears file
    if (fileInput && urlInput) {
        fileInput.addEventListener("change", function () {
            if (fileInput.files && fileInput.files.length > 0) {
                urlInput.value = "";
            }
        });

        urlInput.addEventListener("input", function () {
            if (urlInput.value.trim() !== "") {
                fileInput.value = "";
            }
        });
    }

    function showPreviewAndAnimate(startRequestCallback) {
        // Hide main button
        if (mainButton) {
            mainButton.classList.add("hidden");
        }

        // Show preview container
        if (previewContainer) {
            previewContainer.classList.add("preview-visible");
        }

        // Show loading overlay
        if (loadingOverlay) {
            loadingOverlay.classList.remove("hidden");
        }

        // Hide previous results while recomputing
        if (resultsArea) {
            resultsArea.classList.add("hidden");
        }

        isSubmitting = true;

        // Fake loading delay (~1.5 s), then actually send request
        setTimeout(startRequestCallback, 200);
    }

   function updateResults(predictions) {
    // Order: predictions[0] = Top 1, [1] = Top 2, [2] = Top 3
    const labels = [top1Label, top2Label, top3Label];

    for (let i = 0; i < 3; i++) {
        const pred = predictions[i];
        const labelElem = labels[i];
        if (!labelElem || !pred) continue;

        const pct = (pred.prob * 100).toFixed(2);
        // Only show "Genre (xx.xx%)", no "Top X:" prefix
        labelElem.textContent = `${pred.class_name} (${pct}%)`;
    }

    if (resultsArea) {
        resultsArea.classList.remove("hidden");
    }
}


    form.addEventListener("submit", function (event) {
        // We handle everything via JS
        event.preventDefault();

        if (isSubmitting) {
            return;
        }

        const file = fileInput && fileInput.files && fileInput.files[0];
        const url = urlInput ? urlInput.value.trim() : "";

        if (!file && !url) {
            alert("Please choose a file or enter an image URL first.");
            return;
        }

        // Build form data (file + image_url fields)
        const formData = new FormData(form);

        const startRequest = function () {
            fetch("/api/classify", {
                method: "POST",
                body: formData,
            })
                .then(function (response) {
                    if (!response.ok) {
                        throw new Error("Server returned " + response.status);
                    }
                    return response.json();
                })
                .then(function (data) {
                    if (!data || !Array.isArray(data.predictions)) {
                        throw new Error("Invalid response format.");
                    }
                    updateResults(data.predictions);
                })
                .catch(function (error) {
                    console.error("Classification error:", error);
                    alert("There was an error classifying the image. Please try again.");
                })
                .finally(function () {
                    isSubmitting = false;
                    // Show button again
                    if (mainButton) {
                        mainButton.classList.remove("hidden");
                    }
                    // Hide loading overlay (keep preview visible)
                    if (loadingOverlay) {
                        loadingOverlay.classList.add("hidden");
                    }
                });
        };

        // PRIORITY: if URL is present, ignore file and use URL
        if (url) {
            if (fileInput) {
                fileInput.value = "";
            }
            if (previewImg) {
                previewImg.src = url;
            }
            showPreviewAndAnimate(startRequest);
            return;
        }

        // Otherwise, use the file (we know file is truthy here)
        if (file) {
            if (urlInput) {
                urlInput.value = "";
            }
            const reader = new FileReader();
            reader.onload = function (e) {
                if (previewImg && e.target && typeof e.target.result === "string") {
                    previewImg.src = e.target.result;
                }
                showPreviewAndAnimate(startRequest);
            };
            reader.readAsDataURL(file);
        }
    });
});
