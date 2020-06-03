// below function operates when DOM Content is loaded
function onStart () {

	//	Initialising variables from form to be posted
	let addEmpl;
	let addJob;
	let addDesc;
	let addURL;

	$("#submit").click(async function () {

		addEmpl = $("#addEmpl").val();
		addJob = $("#addJob").val();
		addDesc = $("#addDesc").val();
		addURL = $("#addURL").val();

		//	Error catching for inputs
		try {

			if (addEmpl == "" && document.getElementById("addEmpl").placeholder == "Enter an Employer..." || addJob == "" || addDesc == "" || addURL == "") throw "Empty";
			if (testUrl(addURL) == false) throw "Invalid URL";
			if (addDesc.split(" ").length > 200) throw "Too Long";

			// Taking value of automatically filled placeholder when logged in
			if (document.getElementById("addEmpl").placeholder != "Enter an Employer...") {

				if (addEmpl == "") {

					addEmpl = document.getElementById("addEmpl").placeholder;

				}

			}
			document.getElementById("submit").value = "Submit";

			//	Setting posting image to profile image
			let addImg = document.getElementById("hiddenImg").innerHTML;

			//	Creating random profile pic for users not signed in
			let myCol = "#";
			let chars = ["A", "B", "C", "D", "E", "F", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
			while (myCol.length != 7) {

				myCol += chars[Math.floor(Math.random() * chars.length)];

			}

			if (addImg == "") {

				addImg = "<svg width='55px' height='38px'> <rect x='10' y='0' width='35' height='35' style='fill:" + myCol + ";'/> </svg>";

			}

			// Removing message of no postings
			document.getElementById("emptyMsg").style.display = "none";
			document.body.style.backgroundColor = "#fcfdff";

			// Adding verified tag for users signed in
			if (document.getElementById("Gsignout").style.display == "block") {

				addEmpl += (" (verified)");

			}

			//	Posting entities
			$.post("./add", {addEmpl:addEmpl , addJob:addJob, addDesc:addDesc, addURL:addURL, addImg:addImg});

			//	Fetching entities
			let descresp = await fetch("./descList");
			let jobresp = await fetch("./jobList");
			let linkresp = await fetch("./linkList");
			let imgresp = await fetch("./imgList");
			let empresp = await fetch("./empList");
			let empbody = await empresp.text();
			let descbody = await descresp.text();
			let jobbody = await jobresp.text();
			let linkbody = await linkresp.text();
			let imgbody = await imgresp.text();
			let empList = JSON.parse(empbody);
			let descriptionsList = JSON.parse(descbody);
			let jobList = JSON.parse(jobbody);
			let linkList = JSON.parse(linkbody);
			let imgList = JSON.parse(imgbody);

			document.getElementById("postedJobs").innerHTML = "<div>";

			// Appending fetched lists to the body.
			for(let i = 0; i < empList.length; i++) {

				document.getElementById("postedJobs").innerHTML += "<a href='" + linkList[i] + "' class='jobLink'> <div class='jobsEntries'> <b> " + imgList[i] + empList[i] + "</b> - " + jobList[i] + "<br> <span>" + descriptionsList[i] + "</span> </div> </a>";

			}

			document.getElementById("postedJobs").innerHTML += "</div>";

			//	Reset Submit form
			document.getElementById("submitEntity").reset();

			// 	Remove form from screen
			toggle_hidden("postJob");

		}
		catch(err) {

			alert("At least one input is: " + err);

		}

	});

	// Animated section revealing an about section and adjusting other screen elements
	$("#aboutButton").click(async function () {

		if (document.getElementById("aboutButton").value == "About Janus") {

			document.getElementById("aboutButton").value = "Close";
			$(document.getElementById("aboutCol")).animate({width: "toggle"});
			$(document.getElementById("postedJobs")).animate({marginLeft: "24%"});

		}
		else {

			document.getElementById("aboutButton").style.display = "inline-block";
			document.getElementById("aboutButton").value = "About Janus";
			$(document.getElementById("aboutCol")).animate({width: "toggle"});
			$(document.getElementById("postedJobs")).animate({marginLeft: "0%"});

		}

	});

	//	Searcing for specific jobs or titles
	$("#searchbtn").click(async function () {

		try {

			let descresp = await fetch("./descList");
			let jobresp = await fetch("./jobList");
			let linkresp = await fetch("./linkList");
			let imgresp = await fetch("./imgList");
			let empresp = await fetch("./empList");
			let empbody = await empresp.text();
			let descbody = await descresp.text();
			let jobbody = await jobresp.text();
			let linkbody = await linkresp.text();
			let imgbody = await imgresp.text();
			let empList = JSON.parse(empbody);
			let descriptionsList = JSON.parse(descbody);
			let jobList = JSON.parse(jobbody);
			let linkList = JSON.parse(linkbody);
			let imgList = JSON.parse(imgbody);

			// Keyword creation then used in searching jobs
			let keyword = document.getElementById("headSearch").value;

			document.getElementById("postedJobs").innerHTML = "<div>";

			for(let i = 0; i < empList.length; i++) {

				if(empList[i].toUpperCase().includes(keyword.toUpperCase())) {

					document.getElementById("postedJobs").innerHTML += "<a href='" + linkList[i] + "' class='jobLink'> <div class='jobsEntries'> <b> " + imgList[i] + empList[i] + "</b> - " + jobList[i] + "<br> <span>" + descriptionsList[i] + "</span> </div> </a>";

				}
				else if(jobList[i].toUpperCase().includes(keyword.toUpperCase())) {

					document.getElementById("postedJobs").innerHTML += "<a href='" + linkList[i] + "' class='jobLink'> <div class='jobsEntries'> <b> " + imgList[i] + empList[i] + "</b> - " + jobList[i] + "<br> <span>" + descriptionsList[i] + "</span> </div> </a>";

				}

			}

			document.getElementById("postedJobs").innerHTML += "</div>";

			document.getElementById("headSearch").value = "";

		} catch (err) {

			alert(err);

		}

	});

}

//	Resuable function to toggle the height of a hidden element
function toggle_hidden (div) {

	let elem = document.getElementById(div);
	$(elem).animate({height: "toggle"});

	if (div == "postJob") {

		if (document.getElementById("newJob").innerHTML == "Cancel") {

			document.getElementById("newJob").innerHTML = "Post a Job";

		}
		else {

			document.getElementById("newJob").innerHTML = "Cancel";

		}

	}

}

// Creation of custom placeholder if user is logged in
function loggedCheck () {

	if ($("#Gsignout").is(":visible")) {

		let name = document.getElementById("masthead").innerHTML;
		let n = name.indexOf("Logged in as");
		let autoEmpl = name.substring(n + 13, name.length);
		document.getElementById("addEmpl").placeholder = autoEmpl;

	}
	else {

		document.getElementById("addEmpl").placeholder = "Enter an Employer...";

	}

}

//	On load fetches for when user first visits page
window.addEventListener("load", async function () {

	try {

		let descresp = await fetch("./descList");
		let jobresp = await fetch("./jobList");
		let linkresp = await fetch("./linkList");
		let imgresp = await fetch("./imgList");
		let empresp = await fetch("./empList");
		let empbody = await empresp.text();
		let descbody = await descresp.text();
		let jobbody = await jobresp.text();
		let linkbody = await linkresp.text();
		let imgbody = await imgresp.text();
		let empList = JSON.parse(empbody);
		let descriptionsList = JSON.parse(descbody);
		let jobList = JSON.parse(jobbody);
		let linkList = JSON.parse(linkbody);
		let imgList = JSON.parse(imgbody);

		document.getElementById("postedJobs").innerHTML = "<div>";

		for(let i = 0; i < empList.length; i++) {

			document.getElementById("postedJobs").innerHTML += "<a href='" + linkList[i] + "' class='jobLink'> <div class='jobsEntries'> <b> " + imgList[i] + empList[i] + "</b> - " + jobList[i] + "<br> <span>" + descriptionsList[i] + "</span> </div> </a>";

		}

		document.getElementById("postedJobs").innerHTML += "</div>";

		if(document.getElementById("postedJobs").innerHTML == "<div></div>") {

			document.getElementById("emptyMsg").style.display = "block";
			document.body.style.backgroundColor = "#e0e0e0";

		}

	} catch (err) {

		alert(err);

	}

});

// Mobile compatibility, toggling header if screen too small
window.addEventListener("resize", function (e) {

	let width = e.target.outerWidth;
	if(width < 450) {

		document.getElementById("aboutButton").style.display = "none";
		document.getElementById("headSearch").style.display = "none";
		document.getElementById("searchbtn").style.display = "none";
		document.getElementById("newJob").style.display = "none";
		document.getElementById("Gsignin").style.display = "none";
		document.getElementById("Gsignout").style.display = "none";

	}
	if(width > 450) {

		document.getElementById("aboutButton").style.display = "block";
		document.getElementById("headSearch").style.display = "block";
		document.getElementById("searchbtn").style.display = "block";
		document.getElementById("newJob").style.display = "block";
		if(document.getElementById("masthead").innerHTML.includes("Logged")) {

			document.getElementById("Gsignout").style.display = "block";

		}
		else {

			document.getElementById("Gsignin").style.display = "block";

		}

	}

});

//	Burger menu to toggle header visibility, may be required after auto removal on mobile
function revealMenu () {

	if(document.getElementById("aboutButton").style.display == "none") {

		document.getElementById("aboutButton").style.display = "block";
		document.getElementById("headSearch").style.display = "block";
		document.getElementById("searchbtn").style.display = "block";
		document.getElementById("newJob").style.display = "block";
		if(document.getElementById("masthead").innerHTML.includes("Logged")) {

			document.getElementById("Gsignout").style.display = "block";

		}
		else {

			document.getElementById("Gsignin").style.display = "block";

		}

	}
	else {

		if(document.getElementById("newJob").innerHTML == "Cancel") {

			toggle_hidden("postJob");

		}
		document.getElementById("aboutButton").style.display = "none";
		document.getElementById("headSearch").style.display = "none";
		document.getElementById("searchbtn").style.display = "none";
		document.getElementById("newJob").style.display = "none";
		document.getElementById("Gsignin").style.display = "none";
		document.getElementById("Gsignout").style.display = "none";

	}

}

//	Google sign in with OATH 2.0 API and layout changing
function onSignin (googleUser) {

	let profile = googleUser.getBasicProfile();

	document.getElementById("hiddenImg").innerHTML = "<img src='" + profile.getImageUrl() + "' height='35' width='35' hspace='10'>";

	let id_token = googleUser.getAuthResponse().id_token;
	if(id_token) {

		document.getElementById("Gsignin").style.display = "none";
		document.getElementById("Gsignout").style.display = "block";
		document.getElementById("masthead").innerHTML += "Logged in as " + profile.getName();

	}
	loggedCheck();

}

//	Google sign out and restoring screen to logged out layout
function signOut () {

	let auth2 = gapi.auth2.getAuthInstance();
	auth2.signOut();
	document.getElementById("Gsignin").style.display = "block";
	document.getElementById("Gsignout").style.display = "none";
	document.getElementById("postJob").style.display = "none";
	document.getElementById("newJob").innerHTML = "Post a Job";
	document.getElementById("masthead").innerHTML = "<div id='masthead'> <table style='width:100%'> <tr> <th style='width:10%'> <button id='menubutton' class='transparentButton'> <ion-icon name='reorder' size='large' style='color:white; zoom:1.6;' onclick='revealMenu();'></ion-icon> </button> </th> <th style='width:90%'> <a class='navbar-brand' href='#' id='headerTitle'><img src='IMG_3383.PNG' id='logo' alt='logo'></a> <br></th> </tr> </table> </div>";
	document.getElementById("hiddenImg").innerHTML = "";

}

//	Checking inputted URL structure in form
function testUrl (addUrl) {

	let format = /(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?/;
	return format.test(addUrl);

}

// Activating function once DOM content has been loaded
document.addEventListener("DOMContentLoaded", function () {

	onStart();

});
