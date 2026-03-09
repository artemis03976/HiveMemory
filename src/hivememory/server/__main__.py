"""HiveMemory Server 启动入口"""

import uvicorn


def main():
    uvicorn.run(
        "hivememory.server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
