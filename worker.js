const LLAMA_MODEL = {
  name: "meta/llama3-405b-instruct-maas",
  region: "us-central1",
};

const CLAUDE_MODELS = {
  "claude-3-opus": {
    vertexName: "claude-3-opus@20240229",
    region: "us-east5",
  },
  "claude-3-sonnet": {
    vertexName: "claude-3-sonnet@20240229",
    region: "us-central1",
  },
  "claude-3-haiku": {
    vertexName: "claude-3-haiku@20240307",
    region: "us-central1",
  },
  "claude-3-5-sonnet": {
    vertexName: "claude-3-5-sonnet@20240620",
    region: "us-east5",
  },
  "claude-3-opus-20240229": {
    vertexName: "claude-3-opus@20240229",
    region: "us-east5",
  },
  "claude-3-sonnet-20240229": {
    vertexName: "claude-3-sonnet@20240229",
    region: "us-central1",
  },
  "claude-3-haiku-20240307": {
    vertexName: "claude-3-haiku@20240307",
    region: "us-central1",
  },
  "claude-3-5-sonnet-20240620": {
    vertexName: "claude-3-5-sonnet@20240620",
    region: "us-east5",
  },
};

addEventListener("fetch", (event) => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  let headers = new Headers({
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  });
  if (request.method === "OPTIONS") {
    return new Response(null, { headers });
  } else if (request.method === "GET") {
    return createErrorResponse(405, "invalid_request_error", "GET method is not allowed");
  }

  const authHeader = request.headers.get("Authorization");
  const apiKey = request.headers.get("x-api-key");
  
  if ((!authHeader || !authHeader.startsWith("Bearer ")) && (!API_KEY || API_KEY !== apiKey)) {
    return createErrorResponse(401, "authentication_error", "Invalid or missing authentication");
  }

  const signedJWT = await createSignedJWT(CLIENT_EMAIL, PRIVATE_KEY);
  const [token, err] = await exchangeJwtForAccessToken(signedJWT);
  if (token === null) {
    console.log(`Invalid jwt token: ${err}`);
    return createErrorResponse(500, "api_error", "Invalid authentication credentials");
  }

  try {
    const url = new URL(request.url);
    const normalizedPathname = url.pathname.replace(/^(\/)+/, '/');
    switch(normalizedPathname) {
      case "/v1/chat/completions":
        return handleLlamaChatCompletions(request, token);
      case "/v1/v1/messages":
      case "/v1/messages":
      case "/messages":
        return handleClaudeMessages(request, token);
      default:
        return createErrorResponse(404, "not_found_error", "Not Found");
    }
  } catch (error) {
    console.error(error);
    return createErrorResponse(500, "api_error", "An unexpected error occurred");
  }
}

async function handleLlamaChatCompletions(request, api_token) {
  let payload;

  try {
    payload = await request.json();
  } catch (err) {
    return createErrorResponse(400, "invalid_request_error", "The request body is not valid JSON.");
  }

  const stream = payload.stream || false;
  payload.model = LLAMA_MODEL.name;
  payload.messages = processMessages(payload.messages);

  const url = `https://${LLAMA_MODEL.region}-aiplatform.googleapis.com/v1beta1/projects/${PROJECT}/locations/${LLAMA_MODEL.region}/endpoints/openapi/chat/completions`;

  return fetchAndStreamResponse(url, payload, api_token, stream);
}

async function handleClaudeMessages(request, api_token) {
  const anthropicVersion = request.headers.get('anthropic-version');
  if (anthropicVersion && anthropicVersion !== '2023-06-01') {
    return createErrorResponse(400, "invalid_request_error", "API version not supported");
  }

  let payload;
  try {
    payload = await request.json();
  } catch (err) {
    return createErrorResponse(400, "invalid_request_error", "The request body is not valid JSON.");
  }

  payload.anthropic_version = "vertex-2023-10-16";

  if (!payload.model) {
    return createErrorResponse(400, "invalid_request_error", "Missing model in the request payload.");
  } else if (!CLAUDE_MODELS[payload.model]) {
    return createErrorResponse(400, "invalid_request_error", `Model \`${payload.model}\` not found.`);
  }

  const stream = payload.stream || false;
  const model = CLAUDE_MODELS[payload.model];
  const url = `https://${model.region}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${model.region}/publishers/anthropic/models/${model.vertexName}:streamRawPredict`;
  delete payload.model;

  return fetchAndStreamResponse(url, payload, api_token, stream);
}

async function fetchAndStreamResponse(url, payload, api_token, stream) {
  let response, contentType;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${api_token}`
      },
      body: JSON.stringify(payload)
    });
    contentType = response.headers.get("Content-Type") || "application/json";
  } catch (error) {
    return createErrorResponse(500, "api_error", "Server Error");
  }

  if (stream && contentType.startsWith('text/event-stream')) {
    if (!(response.body instanceof ReadableStream)) {
      return createErrorResponse(500, "api_error", "Server Error");
    }

    const encoder = new TextEncoder();
    const decoder = new TextDecoder("utf-8");
    let buffer = '';
    let { readable, writable } = new TransformStream({
      transform(chunk, controller) {
        let decoded = decoder.decode(chunk, { stream: true });
        buffer += decoded;
        let eventList = buffer.split(/\r\n\r\n|\r\r|\n\n/g);
        if (eventList.length === 0) return;
        buffer = eventList.pop();

        for (let event of eventList) {
          controller.enqueue(encoder.encode(`${event}\n\n`));
        }
      },
    });
    response.body.pipeTo(writable);
    return new Response(readable, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } else {
    try {
      let data = await response.text();
      return new Response(data, {
        status: response.status,
        headers: {
          "Content-Type": contentType,
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error) {
      return createErrorResponse(500, "api_error", "Server Error");
    }
  }
}

function processMessages(messages) {
  let userFound = false;
  let assistantIndex = -1;

  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'user') {
      userFound = true;
      break;
    }
    if (messages[i].role === 'assistant' && assistantIndex === -1) {
      assistantIndex = i;
      break;
    }
  }

  if (assistantIndex !== -1 && !userFound) {
    messages.splice(assistantIndex, 0, {
      role: 'user',
      content: 'Start'
    });
  }

  return messages;
}

function createErrorResponse(status, errorType, message) {
  const errorObject = { type: "error", error: { type: errorType, message: message } };
  return new Response(JSON.stringify(errorObject), {
    status: status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

async function createSignedJWT(email, pkey) {
  pkey = pkey.replace(/-----BEGIN PRIVATE KEY-----|-----END PRIVATE KEY-----|\r|\n|\\n/g, "");
  let cryptoKey = await crypto.subtle.importKey(
    "pkcs8",
    str2ab(atob(pkey)),
    {
      name: "RSASSA-PKCS1-v1_5",
      hash: { name: "SHA-256" },
    },
    false,
    ["sign"]
  );

  const authUrl = "https://www.googleapis.com/oauth2/v4/token";
  const issued = Math.floor(Date.now() / 1000);
  const expires = issued + 600;

  const header = {
    alg: "RS256",
    typ: "JWT",
  };

  const payload = {
    iss: email,
    aud: authUrl,
    iat: issued,
    exp: expires,
    scope: "https://www.googleapis.com/auth/cloud-platform",
  };

  const encodedHeader = urlSafeBase64Encode(JSON.stringify(header));
  const encodedPayload = urlSafeBase64Encode(JSON.stringify(payload));

  const unsignedToken = `${encodedHeader}.${encodedPayload}`;

  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    cryptoKey,
    str2ab(unsignedToken)
  );

  const encodedSignature = urlSafeBase64Encode(signature);
  return `${unsignedToken}.${encodedSignature}`;
}

async function exchangeJwtForAccessToken(signed_jwt) {
  const auth_url = "https://www.googleapis.com/oauth2/v4/token";
  const params = {
    grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
    assertion: signed_jwt,
  };

  const r = await fetch(auth_url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: Object.entries(params)
      .map(([k, v]) => k + "=" + v)
      .join("&"),
  }).then((res) => res.json());

  if (r.access_token) {
    return [r.access_token, ""];
  }

  return [null, JSON.stringify(r)];
}

function str2ab(str) {
  const buffer = new ArrayBuffer(str.length);
  let bufferView = new Uint8Array(buffer);
  for (let i = 0; i < str.length; i++) {
    bufferView[i] = str.charCodeAt(i);
  }
  return buffer;
}

function urlSafeBase64Encode(data) {
  let base64 = typeof data === "string" ? btoa(encodeURIComponent(data).replace(/%([0-9A-F]{2})/g, (match, p1) => String.fromCharCode(parseInt("0x" + p1)))) : btoa(String.fromCharCode(...new Uint8Array(data)));
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
