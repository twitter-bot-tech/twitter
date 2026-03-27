#!/usr/bin/env python3
"""
一次性 Google OAuth2 认证脚本
运行后会输出一个授权 URL，在浏览器中打开并授权，
然后把回调 URL 粘贴回来，脚本会把 token 保存到 token.json
"""
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

SCRIPT_DIR = Path(__file__).parent
SECRETS_FILE = SCRIPT_DIR / "client_secrets.json"
TOKEN_FILE = SCRIPT_DIR / "token.json"


def main():
    flow = InstalledAppFlow.from_client_secrets_file(str(SECRETS_FILE), SCOPES)

    # 生成授权 URL（不自动打开浏览器）
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    print("\n" + "="*60)
    print("请在浏览器中打开以下链接并授权：")
    print("="*60)
    print(auth_url)
    print("="*60)
    print("\n授权后，Google 会显示一个授权码，请将其粘贴到下方：")

    code = input("授权码: ").strip()
    flow.fetch_token(code=code)

    creds = flow.credentials
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print(f"\n✅ token 已保存到: {TOKEN_FILE}")


if __name__ == "__main__":
    main()
