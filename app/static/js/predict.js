// Update file button labels when files are selected
function updateFileButton(inputId) {
    const input = document.getElementById(inputId);
    const button = input.parentElement.querySelector('.file-button');
    
    input.addEventListener('change', function() {
        if (this.files.length > 0) {
            button.textContent = this.files[0].name;
            button.classList.add('has-file');
        } else {
            button.textContent = 'Choose file';
            button.classList.remove('has-file');
        }
    });
}

updateFileButton('flair');
updateFileButton('t1');
updateFileButton('t1ce');
updateFileButton('t2');

document.getElementById("runBtn").onclick = async function() {
    const flair = document.getElementById("flair").files[0];
    const t1 = document.getElementById("t1").files[0];
    const t1ce = document.getElementById("t1ce").files[0];
    const t2 = document.getElementById("t2").files[0];
    const model = document.getElementById("model").value;
    const sliceIndex = document.getElementById("slice_index").value;

    if (!flair || !t1 || !t1ce || !t2) {
        alert("⚠️ Please upload all four MRI modalities (FLAIR, T1, T1CE, T2)");
        return;
    }

    // Show loading state
    document.getElementById("loadingState").classList.remove("hidden");
    document.getElementById("resultsSection").classList.add("hidden");

    const fd = new FormData();
    fd.append("flair", flair);
    fd.append("t1", t1);
    fd.append("t1ce", t1ce);
    fd.append("t2", t2);
    fd.append("model", model);
    if (sliceIndex) fd.append("slice_index", sliceIndex);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: fd
        });

        const data = await res.json();

        // Hide loading state
        document.getElementById("loadingState").classList.add("hidden");

        if (!res.ok) {
            document.getElementById("resultsSection").classList.remove("hidden");
            document.getElementById("debug").innerText = JSON.stringify(data, null, 2);
            return;
        }

        // Show results
        document.getElementById("resultsSection").classList.remove("hidden");

        // Preview image
        const prev = document.getElementById("preview_img");
        prev.src = "data:image/png;base64," + data.preview_b64;

        // Mask image
        const mask = document.getElementById("mask_img");
        mask.src = "data:image/png;base64," + data.mask_b64;

        // Debug info
        document.getElementById("debug").innerText = JSON.stringify(data.debug, null, 2);
    } catch (error) {
        document.getElementById("loadingState").classList.add("hidden");
        alert("❌ Error: " + error.message);
    }
};