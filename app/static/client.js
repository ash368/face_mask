var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select an image to style!");

  el("analyze-button").innerHTML = "Masking..";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.responseType="blob";

  xhr.onload = function(e) {
    if (this.readyState === 4) {
       const blobUrl = URL.createObjectURL(e.target.response);
       el("image-picked").src = blobUrl;
        }
    el("analyze-button").innerHTML = "Mask-photo";
    // el('result-label').innerHTML = '<a>To download image 📥<br> <br>for pc/laptop users 🖥️: by right clicking the mouse on image and choose "Save image as..." <br><br> for mobile users 📱: long press on the image and choose "Download image" option</a>'
      
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  // el("analyze-button").innerHTML = "Masking..";
  xhr.send(fileData);
}

function down1() {
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("GET", `${loc.protocol}//${loc.hostname}:${loc.port}/download`
    ,true);
  xhr.responseType="blob";
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      const blobUrl = window.URL.createObjectURL(e.target.response);
      const link = document.createElement('a');
      link.hidden=true;
      link.download='mask.png'
      link.href = blobUrl;
      link.text = 'downloading....';
      document.body.appendChild(link);
     

      link.click()
      setTimeout(() => {
    // For Firefox it is necessary to delay revoking the ObjectURL
      window.URL.revokeObjectURL(blobURL);
    }, 10);

      URL.revokeObjectURL(blobUrl);
      link.remove(); 
    }
  };
  var fileData = new FormData();
  xhr.send(fileData);
}
