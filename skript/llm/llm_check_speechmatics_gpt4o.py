from llm_check_base import run

run(
    stt_filter='speechmatics',
    llm_filter='gpt-4o',
    title='Speechmatics + GPT-4o',
    out_file='llm_check_speechmatics_gpt4o.md',
)
