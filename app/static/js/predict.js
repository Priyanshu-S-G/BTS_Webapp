document.getElementById("runBtn").onclick = async function() {

    const flair = document.getElementById("flair").files[0];
    const t1 = document.getElementById("t1").files[0];
    const t1ce = document.getElementById("t1ce").files[0];
    const t2 = document.getElementById("t2").files[0];
    const model = document.getElementById("model").value;
    const sliceIndex = document.getElementById("slice_index").value;

    if (!flair || !t1 || !t1ce || !t2) {
        alert("Please upload all four modalities.");
        return;
    }

    const fd = new FormData();
    fd.append("flair", flair);
    fd.append("t1", t1);
    fd.append("t1ce", t1ce);
    fd.append("t2", t2);
    fd.append("model", model);
    if (sliceIndex) fd.append("slice_index", sliceIndex);

    document.getElementById("debug").innerText = "Processing...";

    const res = await fetch("/predict", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    if (!res.ok) {
        document.getElementById("debug").innerText = JSON.stringify(data, null, 2);
        return;
    }

    // preview image
    const prev = document.getElementById("preview_img");
    prev.src = "data:image/png;base64," + data.preview_b64;
    prev.style.display = "block";

    // mask image
    const mask = document.getElementById("mask_img");
    mask.src = "data:image/png;base64," + data.mask_b64;
    mask.style.display = "block";

    document.getElementById("debug").innerText = JSON.stringify(data.debug, null, 2);
};
