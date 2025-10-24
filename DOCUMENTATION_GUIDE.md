# PixelFlow Documentation Guide
*Creating Human-Centric Computer Vision Documentation*

## Philosophy: Beyond Technical Specs

Great documentation doesn't just explain **what** your code does—it explains **why it matters** and **how it transforms work**. PixelFlow bridges the gap between complex computer vision and practical applications, so our documentation must do the same.

### Core Principles

1. **Lead with Problems, Not Solutions**: Start with real-world challenges before diving into technical implementations
2. **Visual First**: Use images, diagrams, and before/after examples to show impact
3. **Progressive Disclosure**: Simple concepts → advanced configurations → expert optimizations
4. **Human Context**: Every feature should connect to a human need or business outcome

## Documentation Structure Templates

### 1. Module Overview Pattern

Instead of listing functions, frame modules around **human needs**:

```markdown
# Privacy & Compliance Tools
*Protect sensitive information while maintaining analytical value*

Modern applications often need to analyze visual data while respecting privacy. Whether you're processing retail footage, medical imagery, or security cameras, these tools let you extract insights while protecting individual privacy.

## Quick Solutions
- **Blur faces in customer analytics**: `blur()` - Maintain shopping behavior insights while protecting customer identity
- **Redact sensitive areas**: `pixelate()` - Obscure license plates, documents, or restricted zones
- **Smart masking**: `mask()` - Highlight important regions while hiding irrelevant details

[Continue with technical details after establishing the why...]
```

### 2. Function Documentation Template

Transform technical docstrings into problem-solution narratives:

```markdown
## Blur Detection Areas
*Protect privacy without losing analytical value*

### Why This Matters
Security cameras capture valuable business intelligence, but they also record personal information that requires protection. Traditional approaches either blur everything (losing insights) or nothing (violating privacy). Smart blurring maintains the analytical value while protecting individuals.

### Quick Start
```python
import pixelflow as pf
import cv2

# Load your image and run detection
image = cv2.imread("security_camera_feed.jpg")
detections = pf.from_ultralytics(model.predict(image))

# Protect privacy while keeping insights
privacy_protected = pf.annotate.blur(image, detections)
```

### Advanced Configuration
[Technical details, parameters, edge cases...]

### Performance Notes
- **Real-time capable**: Processes 1080p at 30fps on standard hardware
- **Memory efficient**: In-place processing reduces memory footprint by 60%
- **Adaptive quality**: Automatically adjusts blur intensity based on detection confidence
```

## Content Strategy Guidelines

### A. Problem-First Approach

**❌ Technical First:**
"The `from_ultralytics()` converter transforms YOLOv8 outputs into PixelFlow Detection objects."

**✅ Problem First:**
"Detection models output raw predictions that need standardization across different frameworks. Convert varied model outputs into a unified format with automatic coordinate normalization, confidence filtering, and class mapping."

### B. Factual vs. Marketing Language

**❌ Marketing Language:**
- "Your AI model just detected 47 objects in a busy street scene. Now what?"
- "Battle-tested in production systems processing millions of frames"
- "Reduce accidents by 23% through early warning systems"

**✅ Factual Language:**
- "Detection models output raw coordinates that require visual representation"
- "Designed for production environments with consistent performance"
- "Enable automated incident detection for traffic monitoring systems"

### C. Visual Evidence

Every annotator should include:
- **Before image**: Raw detection output
- **After image**: PixelFlow enhancement
- **Use case context**: Why this transformation matters

### D. Decision Guidance

Help users choose the right tool:

```markdown
## Which Detection Format Should I Use?

**Working with YOLOv8/YOLOv11?** → `pf.from_ultralytics()`
*Most common choice for real-time applications*

**Research with Detectron2?** → `pf.from_detectron2()`
*Best accuracy for complex scenes*

**Using Datamarkin's API?** → `pf.from_datamarkin()`
*Instant deployment, no model management*

**Custom model outputs?** → Build your own converter
*Complete control over the pipeline*
```

## Module-Specific Strategies

### Annotators Module
**Frame as**: Visual enhancement tools for specific business needs

**Key Messages**:
- Transform raw detections into professional visualizations
- Each annotator solves a specific communication challenge
- Performance-optimized for real-time applications

**Example Sections**:
- **Privacy Tools**: `blur`, `pixelate`
- **Security Monitoring**: `box`, `mask`, `zones`
- **Analytics Visualization**: `heatmap`, `motion_trails`
- **Professional Presentation**: `filled_box`, `anchors`

### Detections Module
**Frame as**: The universal translator between AI models and actionable insights

**Key Messages**:
- Any AI model → Standardized format
- Focus on interoperability and consistency
- Eliminate vendor lock-in

**Narrative Arc**:
1. **The Problem**: Every AI framework has different output formats
2. **The Solution**: Universal detection standard
3. **The Benefit**: Write once, work with any model

### Zones Module
**Frame as**: Spatial intelligence for real-world monitoring

**Key Messages**:
- Transform video feeds into structured data
- Automate spatial analysis that currently requires human monitoring
- Scale monitoring beyond human capacity

**Use Case Categories**:
- **Security**: Restricted area monitoring
- **Business Intelligence**: Customer flow analysis
- **Safety Compliance**: PPE detection in work zones
- **Operational Efficiency**: Queue management and bottleneck identification

### Crossings Module
**Frame as**: People counting and flow analysis for business intelligence

**Key Messages**:
- Convert video into actionable business metrics
- Automate manual counting processes
- Understand traffic patterns and trends

## Writing Style Guidelines

### Voice and Tone
- **Confident but not arrogant**: "PixelFlow handles this automatically"
- **Practical over academic**: Focus on solving problems, not showcasing complexity
- **Inclusive**: Write for both CV experts and newcomers
- **Results-oriented**: Emphasize outcomes and value

### Technical Accuracy
- Always test code examples
- Include realistic data, not placeholder values
- Show error handling for common edge cases
- Mention performance characteristics and limitations

### Human Connection
- Use "you" to directly address the reader
- Include specific industry examples
- Acknowledge common frustrations and pain points
- Celebrate the reader's success

## Mintlify-Specific Implementation

### Component Usage
- **Cards**: Group related features by business function
- **Tabs**: Show platform-specific implementations
- **CodeGroups**: Multiple frameworks for the same concept
- **Expandables**: Progressive disclosure of advanced features

### Visual Hierarchy
```markdown
# Business Function (Cards)
## Common Use Cases (Examples)
### Technical Implementation (Steps)
#### Advanced Configuration (Expandables)
```

### Example Structure
```markdown
---
title: "Privacy Protection Tools"
description: "Maintain analytical value while protecting individual privacy in computer vision applications"
---

# Privacy & Compliance
*Transform sensitive visual data into privacy-compliant insights*

<CardGroup cols={2}>
<Card title="Face Protection" icon="eye-slash" href="/privacy/blur">
  Blur faces in retail analytics while maintaining shopping behavior insights
</Card>
<Card title="Document Redaction" icon="file-shield" href="/privacy/pixelate">
  Obscure sensitive text and documents in security footage
</Card>
</CardGroup>

## Why Privacy Protection Matters

[Human context and business justification...]

<Tabs>
<Tab title="Retail Analytics">
[Specific retail use case...]
</Tab>
<Tab title="Healthcare Compliance">
[Healthcare-specific requirements...]
</Tab>
</Tabs>
```

## Success Metrics

Your documentation succeeds when:

1. **Non-experts can get started quickly**: First working example within 5 minutes
2. **Experts find advanced capabilities**: Deep configuration options clearly documented  
3. **Decision-making is clear**: Users know which tool to choose for their specific need
4. **Visual results are obvious**: Before/after examples show clear value
5. **Business value is articulated**: Each feature connects to a measurable outcome

## Content Templates

### Quick Reference Cards
```markdown
<Card title="Real-Time Blur" icon="eye-slash">
**Best for**: Privacy protection in live video feeds
**Use case**: Compliance with privacy regulations
**Integration**: Compatible with standard video pipelines
**Setup time**: Minimal configuration required
</Card>
```

### Decision Trees
```markdown
## Choosing the Right Annotator

**Need to protect privacy?** 
→ Yes: Use `blur()` for faces, `pixelate()` for text
→ No: Continue to visual enhancement

**Need to highlight detections?**
→ Subtle: Use `box()` with thin lines
→ Prominent: Use `filled_box()` for maximum visibility
```

### Performance Context
```markdown
<Note>
**Performance Considerations**: Optimized for real-time video processing. Performance varies by hardware, image resolution, and number of detections. Test with your specific setup for accurate benchmarks.
</Note>
```

## Documentation Standards

### Title Naming Convention
- **Match file or tool name**: Title must equal the file name or tool name being documented
- **Direct mapping**: For `pixelflow/detections/filters.py` use "Filters" or "Detection Filters"
- **Function-specific docs**: Use the function name as the title
- **Avoid abstract titles**: No marketing language or conceptual abstractions in titles

### Required Elements
- **Clear problem statement**: What challenge does this solve?
- **Practical applications**: Where is this commonly used?
- **Working code examples**: Tested, realistic implementations
- **Technical accuracy**: Only include verified performance data

### Prohibited Elements
- **Unsupported metrics**: No percentages, performance claims, or scale numbers without data
- **Marketing language**: Avoid rhetorical questions, superlatives, and promotional tone
- **Unverifiable claims**: No "battle-tested", "millions of users", or similar statements
- **Hypothetical scenarios**: Focus on real applications, not invented use cases

Remember: Every piece of documentation should answer "So what?" while maintaining technical credibility and factual accuracy. Your goal is to inform and enable, not to sell or exaggerate.