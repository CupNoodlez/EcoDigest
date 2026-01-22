import json
import re
import os

def check_mismatch(target_file, start_batch, end_batch):
    # Detect directory of script to find duplicate files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    responses_file = os.path.join(script_dir, target_file)
    prompts_file = os.path.join(script_dir, "prompts_for_ai.txt")
    
    print(f"Checking '{target_file}' against Batches {start_batch}-{end_batch}...")

    # 1. Count responses in the JSON file
    if not os.path.exists(responses_file):
        print(f"Error: {responses_file} not found.")
        return

    response_ids = set()
    try:
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)
            response_count = len(responses_data)
            print(f"Total items in {os.path.basename(responses_file)}: {response_count}")
            
            for item in responses_data:
                if 'id' in item:
                    response_ids.add(item['id'])
    except Exception as e:
        print(f"Error reading {os.path.basename(responses_file)}: {e}")
        return

    # 2. Count items in batches 16-30 of the text file
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found.")
        return
        
    print(f"\nAnalyzing {os.path.basename(prompts_file)} for Batches {start_batch}-{end_batch}...")
    print("-" * 65)
    print(f"{'BATCH':<8} | {'EXPECTED':<10} | {'FOUND':<10} | {'STATUS'}")
    print("-" * 65)
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find batches: === BATCH X/50 (COPY START) === ... === BATCH END ===
    batch_pattern = re.compile(r'=== BATCH (\d+)/50 \(COPY START\) ===(.*?)=== BATCH END ===', re.DOTALL)
    
    matches = batch_pattern.findall(content)
    
    total_prompt_items = 0
    batch_counts = {}
    prompt_ids = set()
    
    # Store order info for analysis
    expected_ordered_ids = []
    expected_id_to_batch = {}
    
    # Sort matches by batch number
    matches.sort(key=lambda x: int(x[0]))
    
    for batch_num_str, batch_content in matches:
        batch_num = int(batch_num_str)
        
        # We only care about the specified batch range
        if start_batch <= batch_num <= end_batch:
            # Find the JSON data part (starts after "DATA:" and [ ... ])
            json_match = re.search(r'DATA:\s*(\[.*\])', batch_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    batch_data = json.loads(json_str)
                    
                    # Get IDs for this specific batch
                    batch_ids = []
                    for item in batch_data:
                        if 'id' in item:
                            bid = item['id']
                            batch_ids.append(bid)
                            # Store order info
                            expected_ordered_ids.append(bid)
                            expected_id_to_batch[bid] = batch_num
                    
                    count = len(batch_ids)
                    
                    # Calculate Found vs Missing for this batch
                    found_in_batch = 0
                    missing_in_current_batch = []
                    
                    for bid in batch_ids:
                        if bid in response_ids:
                            found_in_batch += 1
                        else:
                            missing_in_current_batch.append(bid)
                    
                    # Update global tracking
                    batch_counts[batch_num] = count
                    total_prompt_items += count
                    prompt_ids.update(batch_ids)

                    # Print status
                    status = "✅ OK" if found_in_batch == count else f"❌ MISSING {count - found_in_batch}"
                    print(f"Batch {batch_num:<2} | {count:<10} | {found_in_batch:<10} | {status}")
                    
                    if missing_in_current_batch:
                        print(f"   -> Missing IDs: {missing_in_current_batch}")
                            
                except json.JSONDecodeError as e:
                    print(f"Batch {batch_num}: Error decoding JSON - {e}")
            else:
                 print(f"Batch {batch_num}: Could not find DATA JSON block")

    print("-" * 40)
    print(f"Total items expected from Batches {start_batch}-{end_batch}: {total_prompt_items}")
    print(f"Total items found in response file:      {response_count}")
    print(f"Difference: {total_prompt_items - response_count}")
    print("-" * 40)
    
    # Check for missing IDs
    missing_in_response = prompt_ids - response_ids
    
    if missing_in_response:
        print(f"IDs in Prompts ({start_batch}-{end_batch}) but MISSING in Response file ({len(missing_in_response)}):")
        sample = list(missing_in_response)[:10]
        print(f"Sample missing IDs: {sample}")
        if len(missing_in_response) > 10:
            print("...")
            
    # Check for EXTRA IDs (and location analysis)
    extra_in_response = response_ids - prompt_ids
    
    if extra_in_response:
        print("\n" + "=" * 65)
        print(f"ANALYSIS OF EXTRA IDs found in '{target_file}'")
        print("These IDs exist in the file but were not found in the expected batches.")
        print("-" * 65)
        print(f"{'INDEX':<8} | {'EXTRA ID FOUND':<20} | {'EXPECTED AT THIS POS'}")
        print("-" * 65)
        
        # Build actual ordered list from file
        actual_ordered_ids = [item['id'] for item in responses_data if 'id' in item]
        
        found_count = 0
        for idx, actual_id in enumerate(actual_ordered_ids):
            if actual_id in extra_in_response:
                found_count += 1
                
                # Determine what was expected here
                expected_info = "???"
                if idx < len(expected_ordered_ids):
                    exp_id = expected_ordered_ids[idx]
                    exp_batch = expected_id_to_batch.get(exp_id, "?")
                    
                    if exp_id != actual_id:
                        expected_info = f"{exp_id} (Batch {exp_batch})"
                    else:
                        # If matches, then logic is weird, but fine
                        expected_info = "MATCH (Shouldn't happen for extra id)"
                else:
                    expected_info = "End of Expected List"
                    
                print(f"{idx:<8} | {actual_id:<20} | {expected_info}")
                
        print("-" * 65) 
        
    else:
        # If counts mismatch but no missing IDs, could be duplicates in prompt file
        if total_prompt_items != response_count:
             print("IDs match sets, but counts differ. Checking for duplicates in Prompts file...")
             # (Simple check logic if needed, but the counts usually tell the story)

if __name__ == "__main__":
    # CONFIGURATION: Change these values to check different files
    # Example 1: Check responses_16-30.json
    # check_mismatch("responses_16-30.json", 16, 30)
    
    # Example 2: Check responses_46-50.json
    FILE_NAME = "responses_16-30.json"
    START_BATCH = FILE_NAME.split('_')[1].split('-')[0]
    END_BATCH = FILE_NAME.split('_')[1].split('-')[1].split('.')[0]
    check_mismatch(FILE_NAME, int(START_BATCH), int(END_BATCH))
