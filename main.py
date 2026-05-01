from scripts.load_data import load_data
def main():
    print("Hello from spam-check!")

df = load_data()
print(df.head())

if __name__ == "__main__":
    main()
