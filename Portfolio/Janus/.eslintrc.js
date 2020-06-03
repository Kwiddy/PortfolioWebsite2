module.exports = {
	"env": {
		"browser": true,
		"commonjs": true,
		"es6": true,
		"node": true,
		"jquery": true,
		"jest": true
	},
	"extends": "eslint:recommended",
	"globals": {
		"Atomics": "readonly",
		"SharedArrayBuffer": "readonly"
	},
	"parserOptions": {
		"ecmaVersion": 2018
	},
	"rules": {
		"indent": [
			"error",
			"tab"
		],
		"linebreak-style": [
			"error",
			"unix"
		],
		"quotes": [
			"error",
			"double"
		],
		"semi": [
			"error",
			"always"
			],
			"no-console": "off",
			"no-useless-escape": "off",
			"padded-blocks": "error",
			"padding-line-between-statements": "error",
			"space-before-blocks": "error",
			"space-before-function-paren": "error",
			"space-in-parens": "error",
			"space-infix-ops": "error",
			"no-trailing-spaces": "error",
			"no-multiple-empty-lines": "error",
			"spaced-comment": "error",
			"no-var": "error",
			"vars-on-top": "error",
			"valid-jsdoc": [1, {
				"prefer": {
					"return": "returns"
				}
			}],
			"require-jsdoc": ["error", {
				"require": {
					"FunctionDeclaration": true,
					"MethodDefinition": false,
					"ClassDeclaration": false,
					"ArrowFunctionExpression": true,
					"FunctionExpression": true
				}
			}]
	}
};
