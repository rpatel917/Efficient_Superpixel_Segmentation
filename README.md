# CS-7641 Machine Learning Project

This repository contains the project website for our CS-7641 Machine Learning course project at Georgia Institute of Technology.

## Project Structure

```
.
├── _config.yml              # Jekyll configuration
├── Gemfile                  # Ruby dependencies
├── README.md               # This file
├── index.md                # Home page
├── proposal.md             # Project proposal
├── midterm.md              # Midterm report
├── final.md                # Final report
└── assets/
    └── css/
        └── style.scss      # Custom CSS styling
```

## Setup Instructions

### Local Development

1. **Install Ruby and Bundler**
   ```bash
   # On macOS
   brew install ruby
   
   # On Ubuntu/Debian
   sudo apt-get install ruby-full
   
   # Install Bundler
   gem install bundler
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

3. **Install dependencies**
   ```bash
   bundle install
   ```

4. **Run Jekyll locally**
   ```bash
   bundle exec jekyll serve
   ```

5. **View the site**
   - Open your browser to `http://localhost:4000`

### GitHub Pages Deployment

1. **Create a new GitHub repository**

2. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

3. **Enable GitHub Pages**
   - Go to your repository settings
   - Navigate to "Pages" section
   - Set source
