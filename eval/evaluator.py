import argparse
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import random
random.seed(0)
import re
from openai import OpenAI

client = OpenAI(api_key='<KEY>')
client.api_key = os.environ("OPENAI_API_KEY")
client.base_url = os.environ("OPENAI_API_BASE")


gpt4_open_free_from_cls_prompt = """You are a helpful AI assistant. Now I will give you an answer from model and an answer from label.
All you need to do is focus on these two answers and figure out whether they are referring to the same general class, focusing on the class of object, not attributes such as color, shape, count, spatial or usage.
Respond with 'T' if they refer to the same class and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Remember, the answer from model refers to one of answers from label; even if the answer from the model refers to the subclass of one of answers from label, you should respond 'T'.
Your response should follow the judgement standard of the prompt I give.
Firstly I will give you two examples answer pairs as well as their responses:

Example1:
Input: 1. wooden board, table, pottery 2. This is a 3D model of wooden table. 
Output: T###Both refer to a table.

Example2:
Input: 1. historical vehicle, pioneer wagon, covered wagon, prairie schooner 2. The 3D object model depicts a quaint, old-fashioned cart. The cart is entirely brown, with two sturdy wooden wheels for mobility. The main body of the cart is in the shape of a large, semicircular curve, made of wood and affixed to the wheels. This curved body extends backward, forming a simple, straight tail. Despite its simplicity, it reflects a nostalgic charm and could be used in settings like historical reenactments, antiquated transportation exhibits, or in visual media for a touch of old-world atmosphere. 
Output: F###One refers to a wagon, the other to a truck.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """


gpt4_object_captioning_prompt = """You are a helpful AI assistant. Now I will give you an answer from the model and an answer from the label.
All you need to do is to evaluate these two answers from six aspects separately: 
1. "description": Giving a comprehensive description of the whole 3D model.
2. "color": Demonstrating the color attribute of the whole or the individual objects
3. "shape": Demonstrating the geometric attribute of the whole or the individual objects.
4. "count": Counting the number of the whole or the individual objects.
5. "spatial": Understanding the spatial relations between multiple objects in the 3D model.
6. "usage": Demonstrating the usage or the production purpose of the 3D model.

For any aspect above, you should identify the aspects mentioned in the answer from model and calculate the percentage of these aspects correctly mentioned or partially matched in the answer from label. 
Remember the score is to evaluate how much two answers are match.
When evaluating and comparing each criterion, do not take other criteria into consideration.
Score from 0 to 100. Consider similar concepts and synonyms. 

Your response should include the scores of the six criteria (description, color, shape, count, spatial, and usage score) in the order above.
Remember all scores range from 0 to 100. 
Firstly, I will give you several answer pairs and their corresponding scores. Your response format should follow the example of the prompt I give:


Provide your score (0-100) in the format of below:
'
Scores for each aspects: **[description score, color score, shape score, count score, spatial score, usage score]**
'

For clarity, consider this example:

Label: 
"description": "This 3D model displays a wooden rectangular platform adorned with various items on top. The setup features a sizable dark clay vessel or cauldron positioned on a miniature stand at one end of the platform. At the other end, a table-like structure supports books, miniature vases, and what seems to be a witch's hat. The entire arrangement is decorated with vibrant speckles or splashes of paint in red, green, and blue, creating a magical or whimsical feel. The wood surface shows clear plank marks and a textured finish.",
"color": "The base displays a wooden brown shade, decorated with vibrant speckles in red, green, and blue. The large vessel is dark gray. The table holds brown books, two small vases (one blue and one green), and a dark gray pointed hat. The table also features the same speckled pattern as the wooden platform.",
"shape": "The principal structure is a rectangular wooden platform. It includes a circular pot at one end and a rectangular table at the other.",
"count": "The arrangement includes several items: one large pot, one table, multiple books, at least two small vases, and one pointed hat.",
"spatial": "The clay vessel is situated at one end of the wooden platform. The table, bearing various items, is located at the opposite end. All elements are positioned on the wooden platform.",
"usage": "This 3D model can be used in video games, animations, or virtual settings to craft a scene with a magical or fantasy theme, perhaps associated with witchcraft or alchemy. It can also be used for reading, writing, and eating."

Model: "This model portrays a vivid scene of a cartoon-styled table, overflowing with a variety of objects. The table seems to be in use, showcasing a naturalistic depiction of a cluttered table in a domestic or workspace. Objects on it reflect common items like books, pencils, cups, etc. suggesting its functionality as a piece of furniture where different activities such as reading, writing, or drinking can be performed."
Output: Scores for each aspect: **[35,0,0,30,35,75]**

Now score the following:
Label: {ground_truth}
Model: {model_output}
Output: """

chatgpt_object_captioning_prompt = gpt4_object_captioning_prompt
chatgpt_open_free_from_cls_prompt = gpt4_open_free_from_cls_prompt

GPT_PRICES = {
    # * check https://openai.com/pricing for updated price
    "gpt-3.5-turbo-0613": {
        "price_1k_prompt_tokens": 0.0015,
        "price_1k_completion_tokens": 0.002
    },
    "gpt-3.5-turbo-1106": {
        "price_1k_prompt_tokens": 0.0010,
        "price_1k_completion_tokens": 0.002
    },
    "gpt-4-0613":{
        "price_1k_prompt_tokens": 0.03,
        "price_1k_completion_tokens": 0.06  
    },
    "gpt-4-1106-preview":{
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03
    },
    "gpt-4o":{
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03
    }  
}

class OpenAIOpenFreeFormClsEvaluator():
    def __init__(self, inputs, output_dir, output_file, model_type="gpt-4o"):
        """
        Args:
            inputs: A dictionary containing the results of the evaluation. It contains two keys: "results" and "prompt".
                "prompt": str
                "results": [
                    {
                        "object_id": str,
                        "model_output": str,
                        "ground_truth": str
                    }
                ]
        """
        print("-" * 80)
        print("Initializing OpenAIEvaluator...")
        self.results = inputs['results']# * contains two keys: "results" and "prompt"
        self.inference_prompt = inputs['prompt'] # * used to prompt PointLLM
        self.correct_predictions = 0  
        self.total_predictions = 0 
        self.invalid_responses = 0
        self.response_data = [] # to save all the response data by openaigpt
        self.model_type = model_type
        self.check_model_type()

        self.prompt_tokens = 0
        self.completion_tokens = 0

        # self.default_chat_parameters = {
        #     "model": model_type,
        #     "temperature": 1, 
        #     "top_p": 1, 
        #     "max_tokens": 2048
        # }

        # * price
        self.price_1k_prompt_tokens = GPT_PRICES[model_type]["price_1k_prompt_tokens"]
        self.price_1k_completion_tokens = GPT_PRICES[model_type]["price_1k_completion_tokens"]

        self.gpt_prompt = chatgpt_open_free_from_cls_prompt if "gpt-3.5" in model_type else gpt4_open_free_from_cls_prompt
        self.output_dir = output_dir
        self.output_file = output_file
        self.temp_output_file = self.output_file.replace(".json", "_processed_temp.json")
    
    def check_model_type(self):
        # * warning if not using gpt-4, recommend using gpt-4 for this task
        if "gpt-4" not in self.model_type:
            print(f"[WARNING] You are using {self.model_type} for evaluation. We recommend using gpt-4 for this task.")

    def resume_processing(self):
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.correct_predictions = saved_results["correct_predictions"]
            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")
        
    def remove_temp_file(self):
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        if os.path.exists(processed_results_path):
            os.remove(processed_results_path)
            print("-" * 80)
            print(f"Removed Temporary file {processed_results_path}")

    def parse_gpt_response_evaluate(self, gpt_response):
        gpt_response = gpt_response.strip()

        cls_result = gpt_response[0].upper()
        reason = gpt_response[2:] if len(gpt_response) > 2 else ""

        if cls_result not in ['T', 'F']:
            self.invalid_responses += 1
            return 0, "INVALID", gpt_response

        accuracy = 1 if cls_result == 'T' else 0

        return accuracy, cls_result, reason

    def evaluate_result(self, result):
        object_id = result['object_id']
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]

        gpt_response = client.chat.completions.create(model="gpt-4o", messages=messages) 

        prompt_tokens = gpt_response.usage.prompt_tokens
        completion_tokens = gpt_response.usage.completion_tokens

        gpt_response = gpt_response.choices[0].message.content

        accuracy, cls_result, reason = self.parse_gpt_response_evaluate(gpt_response) # return 0, "INVALID", gpt_response if not valid

        return object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens

    def evaluate(self):

        self.resume_processing()
        
        print('-' * 80)
        print("Starting single-thread evaluation...")
        results = self.results

        try:
            for result in tqdm(results):  
                object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens = self.evaluate_result(result)
                self.correct_predictions += accuracy
                self.total_predictions += 1
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'model_output': model_output,
                    'gpt_cls_result': cls_result,
                    'gpt_reason': reason
                })
            
            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def parallel_evaluate(self, num_workers=20):

        self.resume_processing()
        
        print('-' * 80)
        print("Starting parallel evaluation...")
        results = self.results

        try:
            with Pool(num_workers) as pool:
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    for object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        self.correct_predictions += accuracy
                        self.total_predictions += 1
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += completion_tokens

                        if cls_result == 'INVALID':
                            self.invalid_responses += 1

                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            'gpt_cls_result': cls_result,
                            'gpt_reason': reason
                        })

                        pbar.update()  # update the progress bar

            print("Parallel evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()

        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def save_results(self, is_temp=False):
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
        else:
            accuracy = self.correct_predictions / (self.total_predictions - self.invalid_responses) * 100
        with open(output_path, 'w') as f:
            results_to_save = {
                'inference_prompt': self.inference_prompt,
                'prompt': self.gpt_prompt,
                'accuracy': f"{accuracy:.2f}%",
                'total_predictions': self.total_predictions,
                'correct_predictions': self.correct_predictions,
                'invalid_responses': self.invalid_responses,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'GPT_cost': self.get_costs(),
                'results': self.response_data,
            }
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")
    
    def print_results(self):
        print('-' * 80)
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0 # * no results and get error
        else:
            accuracy = self.correct_predictions / (self.total_predictions - self.invalid_responses) * 100
        print("Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Correct Predictions: {self.correct_predictions}")
        print(f"Invalid Responses: {self.invalid_responses}")
        self.print_costs()
    
    def print_costs(self):
        print(f"Prompt Tokens Price: {self.prompt_tokens * self.price_1k_prompt_tokens / 1000:.2f} USD")
        print(f"Completion Tokens Price: {self.completion_tokens * self.price_1k_completion_tokens / 1000:.2f} USD")
    
    def get_costs(self):
        return self.prompt_tokens * self.price_1k_prompt_tokens / 1000 + self.completion_tokens * self.price_1k_completion_tokens / 1000
        
    
class OpenAIObjectCaptioningEvaluator(OpenAIOpenFreeFormClsEvaluator):
    def __init__(self, inputs, output_dir, output_file, model_type="gpt-4o"):
        super().__init__(inputs, output_dir, output_file, model_type)
        self.gpt_prompt = chatgpt_object_captioning_prompt if "gpt-3.5" in model_type else gpt4_object_captioning_prompt

        self.total_scores = 0

    def resume_processing(self):
        processed_results_path = os.path.join(self.output_dir, self.temp_output_file)
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.total_scores = float(saved_results["total_score"])

            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")

    def parse_gpt_response_evaluate(self, gpt_response, ground_truth):
        """
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """

        # * use regular expression to extract
        pattern = r'(\d*#.*)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        gpt_score = gpt_response_list[0]
        reason = gpt_response_list[1] if len(gpt_response_list) > 1 else ""

        try:
            # * convert to int
            gpt_score = int(gpt_score)
            if gpt_score not in range(101): # * in 0-100
                # * not valid range
                gpt_score = -1
        except ValueError:
            print(f"Error: unale to parse {gpt_response}.")
            gpt_score = -1

        if gpt_score == -1:
            reason = gpt_response
        
        return gpt_score, reason

    def evaluate_result(self, result):
        object_id = result.get('object_id', -1)
        ground_truth = result['ground_truth']
        model_output = result['model_output']

        messages = [{"role": "user", "content": self.gpt_prompt.format(ground_truth=ground_truth, model_output=model_output)}]
        
        gpt_response = client.chat.completions.create(model="gpt-4o", messages=messages) 

        prompt_tokens = gpt_response.usage.prompt_tokens
        completion_tokens = gpt_response.usage.completion_tokens

        gpt_response = gpt_response.choices[0].message.content

        gpt_score, reason = self.parse_gpt_response_evaluate(gpt_response, ground_truth) # return 0, "INVALID", gpt_response if not valid

        return object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens

    def evaluate(self):

        self.resume_processing()
        
        print('-' * 80)
        print("Starting single-thread evaluation...")
        results = self.results

        try:
            for result in tqdm(results):  
                object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens = self.evaluate_result(result)

                self.total_scores += gpt_score if gpt_score != -1 else 0
                self.total_predictions += 1
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                
                if gpt_score == -1:
                    self.invalid_responses += 1

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'model_output': model_output,
                    "gpt_score": gpt_score,
                    'gpt_reason': reason
                })
            
            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()
    
    def parallel_evaluate(self, num_workers=20):

        # self.resume_processing()
        
        print('-' * 80)
        print("Starting parallel evaluation...")
        results = self.results

        try:
            with Pool(num_workers) as pool:
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    for object_id, model_output, ground_truth, gpt_score, reason, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        self.total_scores += gpt_score if gpt_score != -1 else 0
                        self.total_predictions += 1
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += completion_tokens
                        
                        if gpt_score == -1:
                            self.invalid_responses += 1

                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            "gpt_score": gpt_score,
                            'gpt_reason': reason
                        })

                        pbar.update()  # update the progress bar

            print("Parallel evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()

        except (Exception, KeyboardInterrupt) as e:
            print(f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit() 

    def save_results(self, is_temp=False):
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        if self.total_predictions - self.invalid_responses == 0:
            average_score = 0 # * no results and get error
        else:
            average_score = self.total_scores / (self.total_predictions - self.invalid_responses)
        with open(output_path, 'w') as f:
            results_to_save = {
                'inference_prompt': self.inference_prompt,
                'gpt_prompt': self.gpt_prompt,
                'average_score': f"{average_score:.2f}",
                'total_score': f"{self.total_scores:.2f}",
                'total_predictions': self.total_predictions,
                'invalid_responses': self.invalid_responses,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'GPT_cost': self.get_costs(), 
                'results': self.response_data,
            }
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")
    
    def print_results(self):
        print('-' * 80)
        if self.total_predictions - self.invalid_responses == 0:
            average_score = 0 # * no results and get error
        else:
            average_score = self.total_scores / (self.total_predictions - self.invalid_responses)
        print("Results:")
        print(f"Average Score: {average_score:.2f}")
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Invalid Responses: {self.invalid_responses}")
        print(f"Prompt Tokens: {self.prompt_tokens}")
        print(f"Completion Tokens: {self.completion_tokens}")

        self.print_costs()


def start_evaluation(results, output_dir, output_file, eval_type="open-free-form-classification", model_type="gpt-3.5-turbo-0613",
                        parallel=True, num_workers=20):
    """
    Args:
        results: dict or file path to the json file containing the dict
        output_file: the path the final evaluation results to be saved.
    """
    if isinstance(results, str):
        with open(results, 'r') as fp:
            results = json.load(fp)

    if eval_type == "open-free-form-classification":
        evaluator = OpenAIOpenFreeFormClsEvaluator(results, output_dir, output_file, model_type=model_type)
    elif eval_type == "object-captioning":
        evaluator = OpenAIObjectCaptioningEvaluator(results, output_dir, output_file, model_type=model_type)
    else:
        raise NotImplementedError(f"eval_type {eval_type} not supported.")

    if parallel:
        evaluator.parallel_evaluate(num_workers=num_workers)
    else:
        evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, \
                        default="YOUR_RESULTS_JSON_PATH", help="Path to the results file.")
    parser.add_argument("--output_dir", type=str, default="YOUR_OUTPUT_DIR", help="Path to the output directory.")
    parser.add_argument("--model_type", type=str, default="gpt-4o", choices=["gpt-4o", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--parallel", default=False, action="store_true", help="Whether to use parallel evaluation.")
    parser.add_argument("--num_workers", type=int, default=15, help="Number of workers to use for parallel evaluation.")
    parser.add_argument("--eval_type", type=str, choices=["open-free-form-classification", "object-captioning"], default="object-captioning")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_path)

    output_file = os.path.basename(args.results_path).replace(".json", f"_evaluated_{args.model_type}.json")

    # if exists, then exit
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        print(f"[INFO] Evaulated results already exists in {os.path.join(args.output_dir, output_file)}.")
        exit()

    start_evaluation(results=args.results_path, output_dir=args.output_dir, output_file=output_file, eval_type=args.eval_type, model_type=args.model_type, 
                        parallel=args.parallel, num_workers=args.num_workers)
    