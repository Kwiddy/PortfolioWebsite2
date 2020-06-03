const app = require("./app");
const request = require("supertest");

describe("Test the various entities", () => {

	// Testing requests succeed

	test("GET succeeds for / request", () => {

		return request(app).get("/").expect(200);

	});

	test("GET /empList succeeds", (done) => {

		return request(app)
			.get("/empList")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /jobList succeeds", (done) => {

		return request(app)
			.get("/jobList")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /descList succeeds", (done) => {

		return request(app)
			.get("/descList")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /linkList succeeds", (done) => {

		return request(app)
			.get("/linkList")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /imgList succeeds", (done) => {

		return request(app)
			.get("/imgList")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /auth/google", (done) => {

		return request(app)
			.get("/auth/google")
			.expect(302)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	// Testing response types

	test("GET /empList returns JSON", (done) => {

		return request(app)
			.get("/empList")
			.expect("Content-type", /json/)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /jobList returns JSON", (done) => {

		return request(app)
			.get("/jobList")
			.expect("Content-type", /json/)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /descList returns JSON", (done) => {

		return request(app)
			.get("/descList")
			.expect("Content-type", /json/)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /linkList returns JSON", (done) => {

		return request(app)
			.get("/linkList")
			.expect("Content-type", /json/)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("GET /imgList returns JSON", (done) => {

		return request(app)
			.get("/imgList")
			.expect("Content-type", /json/)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	// Testing POSTs

	test("POST /add succeeds", (done) => {

		return request(app)
			.post("/add")
			.expect(200)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	// Testing redirect
	// Google Oauth API will detect 400 error as detailed in app.js if necessary
	test("REDIRECT Google OAUTH", (done) => {

		return request(app)
			.get("/auth/google/callback")
			.expect(302)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

	test("AUTHENTICATE Google OAUTH Profile", (done) => {

		return request(app)
			.get("/auth/google")
			.expect(302)
			.end(function (err) {

				if (err) return done(err);
				done();

			});

	});

});
