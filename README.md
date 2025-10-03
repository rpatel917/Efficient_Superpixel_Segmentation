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
└── final.md                # Final report
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
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/ (root)" folder
   - Click "Save"

4. **Access your site**
   - Your site will be available at: `https://yourusername.github.io/your-repo/`
   - It may take a few minutes for the first deployment

## Customization

### Update Team Information

Edit `index.md` to add your team name and members:

```markdown
**Team Name:** [Your Team Name]

**Team Members:**
- [Member 1]
- [Member 2]
- [Member 3]
```

### Fill in the Proposal

Edit `proposal.md` and replace all placeholder text (marked with brackets `[]`) with your actual project details:

- Project domain and problem
- Dataset information
- Specific methods and algorithms
- Expected results
- Relevant references

### Add Content to Other Pages

- **Midterm Report** (`midterm.md`): Add your progress update, preliminary results, and challenges
- **Final Report** (`final.md`): Add complete results, analysis, and conclusions

### Customize Styling

The site uses Jekyll's default Minima theme with the "classic" skin, which provides a clean academic look out of the box. No custom CSS needed!

## Page Structure

### Proposal Sections
- Introduction
- Problem Definition
- Proposed Methods
- Potential Results and Discussion
- References

### Midterm Report Sections (suggested)
- Progress Summary
- Data Exploration and Preprocessing
- Preliminary Model Results
- Challenges Encountered
- Next Steps

### Final Report Sections (suggested)
- Executive Summary
- Complete Methodology
- Results and Analysis
- Model Comparison
- Conclusions
- Future Work
- Complete References

## Tips for Academic Writing

1. **Be specific**: Replace all `[placeholder]` text with concrete details
2. **Cite sources**: Add proper citations in the References section
3. **Include figures**: Add images by placing them in an `assets/images/` folder and referencing them:
   ```markdown
   ![Description](assets/images/your-image.png)
   ```
4. **Use tables**: Markdown tables are great for comparing results:
   ```markdown
   | Model | Accuracy | F1-Score |
   |-------|----------|----------|
   | LR    | 85.2%    | 0.83     |
   | RF    | 89.7%    | 0.88     |
   ```
5. **Add code snippets**: Use code blocks for important code:
   ````markdown
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100)
   ```
   `   ```

## Color Scheme

The site uses Jekyll Minima's classic theme with a clean, professional academic appearance.

## Troubleshooting

### Jekyll Build Errors

If you encounter build errors:

```bash
# Clear the cache
bundle exec jekyll clean

# Rebuild
bundle exec jekyll build

# Serve again
bundle exec jekyll serve
```

### GitHub Pages Not Updating

- Check the "Actions" tab in your repository for build status
- Ensure all files are committed and pushed
- Wait a few minutes for GitHub to rebuild the site
- Check that your `_config.yml` has correct settings

### Local Server Issues

```bash
# Update bundler
gem update bundler

# Reinstall dependencies
bundle install

# If port 4000 is busy, use a different port
bundle exec jekyll serve --port 4001
```

## Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)
- [Minima Theme](https://github.com/jekyll/minima)

## License

This project is for academic purposes as part of CS-7641 at Georgia Tech.

## Contact

[Your Team Email or GitHub Profile]

---

*Last updated: October 2024*
