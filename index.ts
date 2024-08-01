import {
	OpenAI,
	Metadata,
  Settings,
	Document,
  OpenAIAgent,
  FunctionTool,
  QueryEngineTool,
  VectorStoreIndex,
	LlamaParseReader,
	QdrantVectorStore,
  HuggingFaceEmbedding,
} from "llamaindex";
import fs from "node:fs/promises";
import 'dotenv/config'
import readline from 'readline';

async function main() {
	Settings.llm = new OpenAI({
		apiKey: process.env.OPENAI_API_KEY,
		model: "gpt-3.5-turbo",
	});

	const PARSING_CACHE = "./cache.json"

	const vectorStore = new QdrantVectorStore({
		url: "http://localhost:6333",
	});

	Settings.callbackManager.on("llm-tool-call", (event) => {
		// console.log(event.detail)
	});
	Settings.callbackManager.on("llm-tool-result", (event) => {
		// console.log(event.detail)
	});

	Settings.embedModel = new HuggingFaceEmbedding({
		modelType: "BAAI/bge-small-en-v1.5",
		quantized: false,
	});

	// load cache.json and parse it
	let cache = {};
	let cacheExists = false;

	try {
		await fs.access(PARSING_CACHE, fs.constants.F_OK);
		cacheExists = true;
	} catch (e) {
		console.log("No cache found");
	}
	if (cacheExists) {
		cache = JSON.parse(await fs.readFile(PARSING_CACHE, "utf-8"));
	}

	const filesToParse = [
		"./data/ICS_EUR_Ukraine_29AUG2023_PUBLIC.pdf"
	];

	// load our data, reading only files we haven't seen before
	let documents: Document<Metadata>[] = [];
	const reader = new LlamaParseReader({
		resultType: "markdown",
		language: "en"
	});

	for (let file of filesToParse) {
		if (!cache[file]) {
			documents = documents.concat(await reader.loadData(file));
			cache[file] = true;
		}
	}

	// write the cache back to disk
	await fs.writeFile(PARSING_CACHE, JSON.stringify(cache));

	const index = await VectorStoreIndex.fromDocuments(documents, {
		vectorStores: {
			"TEXT": vectorStore
		}
	});

	const retriever = await index.asRetriever({
		similarityTopK: 10
	});

	const queryEngine = await index.asQueryEngine({
		retriever,
	});

	const sumNumbers = ({ a, b }) => {
		return `${a + b}`;
	};

	const tools = [
		new QueryEngineTool({
			queryEngine: queryEngine,
			metadata: {
				name: "ukraines_strengthens_tool",
				description: `This tool can answer detailed questions about the Ukraine's strengthens.`,
			},
		}),
		FunctionTool.from(sumNumbers, {
			name: "sumNumbers",
			description: "Use this function to sum two numbers",
			parameters: {
				type: "object",
				properties: {
					a: {
						type: "number",
						description: "First number to sum",
					},
					b: {
						type: "number",
						description: "Second number to sum",
					},
				},
				required: ["a", "b"],
			},
		}),
	];

	const agent = new OpenAIAgent({ tools });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const askQuestion = (query) => {
    return new Promise((resolve) => {
      rl.question(query, (answer) => {
        resolve(answer);
      });
    });
  };

  while (true) {
    const question = await askQuestion("Enter your question (or 'exit' to quit): ");
    
    if (question.toLowerCase() === 'exit') {
      console.log("Exiting the program.");
      rl.close();
      break;
    }

    const response = await agent.chat({ message: question });
    console.log("Response:", response.message.content);
    console.log();
  }
}

main().catch(console.error);